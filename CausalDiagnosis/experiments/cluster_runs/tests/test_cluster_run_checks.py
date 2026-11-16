"""The checks of the cluster run order on run folders written here by hand: the latency of the model calls, the pilot of the
thinking rate and its projection of a registered run, the development gate and the runs of a suspended or requeued job."""
import csv
import json
import math
import re
from datetime import datetime, timedelta

import pytest

from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.experiments.cluster_runs import run_lines
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered
from CausalDiagnosis.experiments.cluster_runs import cluster_run_checks as run_checks, development_checks, run_health_checks
from CausalDiagnosis.experiments.table_definitions import training_variants as training
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET

CAP = 8
PILOT = dict(config_name=design.TRAINING_PRESET, starts="development", start_cap=CAP, diagnostics_server=True)
GATE = dict(config_name=design.AGENTS_PRESET, starts="development", start_cap=CAP, diagnostics_server=False)
NUMBERS = {"valid_share": 0.5, "exact_match_alpha": 0.05, "max_harness_share": 0.25, "patience_share": 0.9, "patience_factor": 2,
           "patience_steps": 2, "start_share": 0.5}
REAL_START_CAP = development_checks.start_cap  # bound before the fixture below replaces it
DIAGNOSIS = {"event": "submit_diagnosis", "step": 1, "t_start": 1, "diagnosis": {"c1": 1}}
ORDER = {"event": "work_order_created", "step": 2, "t_start": 2, "component": "c1"}
CLOSE = {"event": "work_order_closed", "step": None, "t_start": 6, "component": "c1"}  # the close of the repair of ORDER
FAILURE = {"event": "failure", "step": None, "t_start": 1, "component": "c1"}


def _registered(tmp_path_factory, monkeypatch, **open_values):
    data = registered.load()
    data["development"].update({**NUMBERS, **open_values})
    data["counting"]["max_unscored_share"] = 0.25
    path = tmp_path_factory.mktemp("registered") / registered.PATH.name
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(registered, "PATH", path)


@pytest.fixture(autouse=True)
def numbers(tmp_path_factory, monkeypatch):
    """A copy of the registered file with every number the checks read set, so the tests read the rules and not the open values,
    and the start cap of the runs fixed at CAP, so that the checks read no data file; the rule of the cap has a test of its own."""
    _registered(tmp_path_factory, monkeypatch)
    monkeypatch.setattr(development_checks, "start_cap", lambda asset_id: CAP)


def _run(root, name, agent="causal", passes=(True, False), top1=None, calls=((1.0,), (2.0,)), ends=None, job=None, gaps=None, cells=None,
         logs=None, **meta):
    """A finished run folder: one episode per entry of passes, with its verdict in scores.csv, the seconds of its model calls
    in llm_calls.json with 100 generated tokens for every second and, when gaps names them, the wall and the monotonic seconds before its tool calls in episode.json. cells
    names further columns of scores.csv with one value per episode, and logs the events of the hidden log of each episode, beside
    an oracle that holds one order due."""
    folder = root / name
    top1 = passes if top1 is None else top1
    rows = []
    for i, passed in enumerate(passes):
        episode = folder / f"seed{i:06d}"
        episode.mkdir(parents=True)
        seconds = calls[i] if i < len(calls) else ()
        (episode / "llm_calls.json").write_text(json.dumps([{"seconds": s, "output_tokens": round(100 * s)} for s in seconds]
                                                           + [{"error": "no reply"}]), encoding="utf-8")
        state = {"think_seconds": [], "think_cycles": []}
        if gaps is not None:
            wall, monotonic = gaps[i] if i < len(gaps) else ((), ())
            state.update({run_health_checks.WALL_GAPS: list(wall), **({} if monotonic is None else {run_health_checks.MONOTONIC_GAPS: list(monotonic)})})
        (episode / "episode.json").write_text(json.dumps(state), encoding="utf-8")
        if logs is not None:
            (episode / "hidden").mkdir()
            log = [{"event": "reset", "step": None, "t_start": 0}, *(logs[i] if i < len(logs) else [])]
            (episode / "hidden" / "log.json").write_text(json.dumps(log), encoding="utf-8")
            (episode / "hidden" / "oracle.json").write_text(json.dumps({"work_orders": [{"component": "c1", "window_end": 9}]}), encoding="utf-8")
        rows.append({"seed": i, "pass": passed, "diagnosis_all_correct": top1[i], "end_reason": (ends or {}).get(i, "finish"),
                     **{column: values[i] for column, values in (cells or {}).items()}})
    with open(folder / "scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {"agent": agent, "training_variant": training.AS_RELEASED, "seeds": list(range(len(passes))), "episodes_planned": len(passes),
              "started": "2026-01-01T00:00:00", "ended": "2026-01-01T01:00:00", **meta}
    (folder / "run.json").write_text(json.dumps(record), encoding="utf-8")
    (folder / "ledger.json").write_text(json.dumps({}), encoding="utf-8")
    # the job and its host stand in the private file of the run (public_record), which the published ledger never holds
    (folder / "job.json").write_text(json.dumps({"starts": [{"job": {"slurm_job_id": None, "slurm_restart_count": None, "host": "node", **(job or {})}}]}),
                                     encoding="utf-8")
    return folder


def test_the_latency_of_the_model_calls_is_printed_per_run_and_pooled(tmp_path, capsys):
    """The latency of every model call is recorded and every grid job prints it per run folder, the count,
    the total, the median, the p90 by the nearest rank and the longest call, a call without seconds left out, and one pooled
    line over the runs, the line a pilot reads."""
    _run(tmp_path, "runA", passes=(True,), calls=((10.5, 70.25, 60.0, 3.0),))
    _run(tmp_path, "runB", passes=(True, True), calls=((5.0,), (5.0,)))
    _run(tmp_path, "reference", agent="oracle", passes=(True,), calls=((),))
    (tmp_path / "aggregate").mkdir()
    assert run_checks.main(["latency", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "CALL_LATENCY runA: 4 model calls, total 143.75 s, median 35.25 s, p90 70.25 s, longest 70.25 s",
        "CALL_LATENCY runB: 2 model calls, total 10.00 s, median 5.00 s, p90 5.00 s, longest 5.00 s",
        "CALL_LATENCY pooled over 2 runs: 6 model calls, total 153.75 s, median 7.75 s, p90 70.25 s, longest 70.25 s"]


def test_the_node_of_every_run_is_printed_with_a_count_per_node(tmp_path, capsys):
    """Every clocked run plays on one node type, and the node of every run
    is read afterwards from its ledger, so a reader sees whether the two H100 nodes split a table."""
    _run(tmp_path, "runA", job={"slurm_job_id": "11", "host": "node001"})
    _run(tmp_path, "runB", job={"slurm_job_id": "11", "host": "node001"})
    other = tmp_path / "other"
    other.mkdir()
    _run(other, "runC", job={"slurm_job_id": "12", "host": "node002"})
    assert run_checks.main(["nodes", str(tmp_path), str(other)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "NODE runA: node001, job 11", "NODE runB: node001, job 11", "NODE runC: node002, job 12",
        "NODES 3 runs on 2 nodes: node001 2, node002 1"]


CALLS = (((1.0, 2.0), (30.0,)), ((3.0, 4.0), (5.0,)), ((6.0,), (7.0,)))  # medians 2, 4 and 6.5 per agent, 4.5 pooled over the calls


def _pilot(root, patience=10, logs=None, horizon=None, **change):
    config = {"failure": {"patience": patience}, **({"horizon": horizon} if horizon is not None else {})}
    for i, (agent, _mcp, _name) in enumerate(design.TRAINING_ROWS):
        _run(root, f"pilot_{agent}_{patience}", agent=agent, calls=CALLS[i], logs=(logs or {}).get(agent, [[ORDER, CLOSE], [ORDER, CLOSE]]),
             **{**PILOT, "config": config, **change.get(agent, {})})


def test_the_rate_is_the_median_of_the_three_medians(tmp_path, capsys):
    """The rate of the pilot: per agent the median of the seconds of its model calls, and think_seconds_per_cycle is the median of the three
    medians, 4.00 here where the pooled median of the calls is 4.50, read from development runs alone."""
    _pilot(tmp_path)
    _run(tmp_path, "test_set", agent="causal", calls=((500.0,), (500.0,)), **{**PILOT, "starts": "sampled"})
    assert run_checks.main(["pilot", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "PILOT think_seconds_per_cycle 4.00: the median of the medians" in out and "500" not in out
    assert out.count("PILOT first order, ") == 3 and out.splitlines()[-1] == "PILOT_OK"


def test_the_pilot_writes_seconds_per_token_as_think_seconds_per_cycle_over_the_median_of_the_three_median_generated_tokens(tmp_path, capsys):
    """The rate of the clock of generated tokens: per agent the median of the generated tokens of its model calls (200, 400 and 650
    here), and seconds_per_token is the think_seconds_per_cycle the pilot measures, 4.00, over the median of the three medians, 400."""
    _pilot(tmp_path)
    assert run_checks.main(["pilot", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "PILOT seconds_per_token 0.01: think_seconds_per_cycle 4.00 over the median 400 of the medians of the generated tokens" in out
    assert 'PILOT measured "seconds_per_token": 0.01; every run of record keeps the value the clock of the asset file declares' in out
    assert "PILOT write" not in out  # the pilot tells no one to change a declared value


def test_the_pilot_refuses_a_missing_or_a_second_run_of_an_agent(tmp_path):
    _pilot(tmp_path)
    _run(tmp_path, "again", agent=design.CD_MCP[0], **{**PILOT, "config": {"failure": {"patience": 10}}})
    with pytest.raises(SystemExit, match=f"2 finished development runs of {re.escape(design.CD_MCP[2])}"):
        run_checks.main(["pilot", str(tmp_path)])
    other = tmp_path / "other"
    _run(other, "one", agent=design.CD_MCP[0], **{**PILOT, "config": {"failure": {"patience": 10}}})
    with pytest.raises(SystemExit, match=f"0 finished development runs of {re.escape(design.TRAINING_ROWS[0][2])}"):
        run_checks.main(["pilot", str(other)])


def test_the_patience_rule_asks_for_each_agent_and_multiplies_by_the_registered_factor(tmp_path, capsys):
    """The patience rule: each agent closes the repair of its first order no later than the first failure in the registered share of
    its episodes with an order due. An agent short of it asks for the patience times the registered factor; the next step, played under the
    same folder with that patience, is read on its own runs; the registered number of steps counts the patience values played, the
    first included, after which the check stops, and the share of episodes that a failure ended before any order is printed in
    every step."""
    late = {design.CD_MCP[0]: [[ORDER, CLOSE], [FAILURE, {**ORDER, "t_start": 3}]]}
    _pilot(tmp_path, patience=10, logs=late)
    assert run_checks.main(["pilot", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"PILOT first order, {design.CD_MCP[2]}: 1 of 2 episodes with an order due" in out
    last = out.splitlines()[-1]
    assert last.startswith(f"PILOT_PATIENCE 20: {design.CD_MCP[2]} stay below the registered share 0.9 with the patience 10") and last.endswith("step 2 of 2")
    _pilot(tmp_path, patience=20)
    assert run_checks.main(["pilot", str(tmp_path)]) == 0 and capsys.readouterr().out.splitlines()[-1] == "PILOT_OK"
    again = tmp_path / "again"
    _pilot(again, patience=10, logs=late)
    _pilot(again, patience=20, logs={design.CD_MCP[0]: [[ORDER, CLOSE], [FAILURE]]}, **{design.CD_MCP[0]: {"ends": {1: development_checks.FAILURE_END}}})
    assert run_checks.main(["pilot", str(again)]) == 1
    out = capsys.readouterr().out
    assert "1 of 2 ended by a failure before any order" in out
    assert out.splitlines()[-1].startswith(f"PILOT_STOP {design.CD_MCP[2]} stay below the registered share 0.9 after 2 steps")


def test_the_pilot_stops_where_the_next_patience_reaches_the_horizon_of_the_preset(tmp_path, capsys):
    """A patience at or above the horizon never fires, the failure rule switched off, so no step of the patience rule may reach
    it, for any asset: with the horizon of the runs at twice the patience the pilot stops in place of asking for the longer one."""
    late = {design.CD_MCP[0]: [[ORDER, CLOSE], [FAILURE, {**ORDER, "t_start": 3}]]}
    _pilot(tmp_path, patience=10, logs=late, horizon=20)
    assert run_checks.main(["pilot", str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines()[-1].startswith(
        f"PILOT_STOP {design.CD_MCP[2]} stay below the registered share 0.9 with the patience 10, and the next patience 20 reaches the horizon 20 ")


def test_an_order_placed_before_the_failure_whose_repair_closes_after_it_is_not_in_time(tmp_path, capsys):
    """One deadline: the failure card fires while an order is open, so the pilot counts an episode in time only when the repair
    of its first order closes no later than the first failure. An order placed before the failure whose repair closes after it
    leaves the agent short of the registered share, and the pilot asks for the longer patience."""
    open_at_failure = [{**ORDER, "t_start": 2}, {**FAILURE, "t_start": 4}, {**CLOSE, "t_start": 7}]
    _pilot(tmp_path, patience=10, logs={design.CD_MCP[0]: [open_at_failure, [ORDER, CLOSE]]})
    assert run_checks.main(["pilot", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    line = next(row for row in out.splitlines() if row.startswith(f"PILOT first order, {design.CD_MCP[2]}: "))
    assert line.split(": ", 1)[1].startswith("1 of 2 ") and "close one no later than the first failure" in line
    assert out.splitlines()[-1].startswith(f"PILOT_PATIENCE 20: {design.CD_MCP[2]} stay below the registered share 0.9")


def test_the_start_cap_is_the_registered_share_of_the_development_starts_rounded_up(tmp_path_factory, monkeypatch, capsys):
    """Amendment 34.16 rule 3: a round and a pilot play the registered share of the development starts of the asset, rounded up
    to a whole start with the share read as an exact fraction, and cap prints that count for the launch line."""
    n = development_checks.development_count(DEFAULT_ASSET)
    assert n > 0
    for share, count in ((0.5, -(-n // 2)), (1, n), (1e-9, 1)):
        _registered(tmp_path_factory, monkeypatch, start_share=share)
        assert REAL_START_CAP(DEFAULT_ASSET) == count, share
    assert run_checks.main(["cap"]) == 0 and capsys.readouterr().out.strip() == str(CAP)


def _block(faulty=()):
    profile = get_profile(DEFAULT_ASSET)
    return json.dumps([next(level for level in profile.levels[c] if level != profile.healthy[c]) if c in faulty else profile.healthy[c]
                       for c in profile.components])


def _faulty_sets(n):
    """n blocks of the default asset, each on a faulty set of its own: every component alone, then the pairs."""
    components = get_profile(DEFAULT_ASSET).components
    sets = [{c} for c in components] + [{a, b} for i, a in enumerate(components) for b in components[i + 1:]]
    return [_block(faulty) for faulty in sets[:n]]


def _gate(root, n=5, ours=None, theirs=None, valid=None, exact=None, seeds=None):
    """The three rows without the MCP on the main table's preset, n episodes each on n faulty sets, so the constant answer is
    right once and the bar of the rule is 4 of 5, with the graded rates of the CausalDiagnosis agent and of Plan-and-Execute, the episodes with a
    diagnosis accepted per row (valid, all of them by default) and the exact match of the CausalDiagnosis agent (all right by default)."""
    blocks = _faulty_sets(n)
    graded = {design.CD: ours or {"graded_top1": ("0.5",) * 5, "graded_pass": ("1.0", "0.0", "1.0", "0.0", "1.0")},
              design.PE: theirs or {"graded_top1": ("0.5", "0.25", "0.5", "0.25", "0.5"), "graded_pass": ("0.0", "1.0", "0.0", "1.0", "1.0")}}
    for row in (design.PE, design.AAT, design.CD):
        k = (valid or {}).get(row, n)
        logs = [[DIAGNOSIS] if i < k else [] for i in range(n)]
        right = (exact or ("True",) * n) if row == design.CD else ("False",) * n
        cells = {"block": blocks, "graded_top1": ("0",) * n, "graded_pass": ("0",) * n, "faulty_set_right": right, **graded.get(row, {})}
        _run(root, f"gate_{row[0]}", agent=row[0], passes=(True,) * n, calls=((1.0,),) * n, cells=cells, logs=logs,
             **{**GATE, **({"seeds": seeds} if seeds and row == design.CD else {})})


def test_the_gate_reads_the_graded_rates_the_valid_share_and_the_exact_match_bar(tmp_path, capsys):
    """The development gate as registered: the graded Top-1 and the graded pass of the CausalDiagnosis agent without the MCP reach those of Plan-and-Execute without it,
    every row without the MCP has a diagnosis accepted in the registered share of its episodes, and the count of exact matches of
    the CausalDiagnosis agent's row reaches the bar of the registered rule, the smallest count that an exact one-sided binomial test at the registered
    level separates from the count of the constant answer: 4 of 5 where the constant answer is right once."""
    _gate(tmp_path)
    _run(tmp_path, "with_mcp", agent=design.CD_MCP[0], passes=(False, False), **{**GATE, "diagnostics_server": True})
    assert run_checks.main(["gate", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [f"GATE graded Top-1: {design.CD[2]} 0.500 over 5, {design.PE[2]} 0.400 over 5",
                         f"GATE graded pass: {design.CD[2]} 0.600 over 5, {design.PE[2]} 0.600 over 5"]
    assert [line for line in lines if line.startswith("GATE valid diagnoses")] == [
        f"GATE valid diagnoses, {row[2]}: 5 of 5, at or above the registered share 0.5" for row in (design.PE, design.AAT, design.CD)]
    assert lines[-2] == (f"GATE exact match, {design.CD[2]}: 5 of 5 against the bar 4 of 5, the smallest count that an exact one-sided "
                         "binomial test at 0.05 separates from the 1 of the constant answer") and lines[-1] == "GATE_PASS"


def test_the_bar_passes_at_its_count_and_misses_one_below_it(tmp_path, capsys):
    _gate(tmp_path, exact=("True",) * 4 + ("False",))
    assert run_checks.main(["gate", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2].startswith(f"GATE exact match, {design.CD[2]}: 4 of 5 against the bar 4 of 5") and lines[-1] == "GATE_PASS"
    other = tmp_path / "other"
    _gate(other, exact=("True",) * 3 + ("False",) * 2)
    assert run_checks.main(["gate", str(other)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2].startswith(f"GATE exact match, {design.CD[2]}: 3 of 5 against the bar 4 of 5") and lines[-1] == "GATE_MISS"


@pytest.mark.parametrize("ours", [{"graded_top1": ("0.25",) * 5}, {"graded_pass": ("0.0",) * 5}])
def test_the_gate_misses_when_either_graded_rate_falls_short(tmp_path, capsys, ours):
    _gate(tmp_path, ours={"graded_top1": ("0.5",) * 5, "graded_pass": ("1.0", "0.0", "1.0", "0.0", "1.0"), **ours})
    assert run_checks.main(["gate", str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "GATE_MISS"


def test_the_valid_share_is_read_on_every_row_without_the_mcp_and_the_exact_match_bar_on_our_row_alone(tmp_path, capsys):
    """A reference design below the registered share of valid diagnoses reads GATE_PORT_SUSPECT with its funnel and fails the
    check, while a reference design whose exact match lies below the bar changes nothing, since the bar reads the CausalDiagnosis agent's row alone."""
    episodes = 5
    _gate(tmp_path, n=episodes, valid={design.AAT: 0})
    assert run_checks.main(["gate", str(tmp_path)]) == 1
    lines = capsys.readouterr().out.splitlines()
    suspect = [line for line in lines if line.startswith("GATE_PORT_SUSPECT")]
    assert len(suspect) == 1 and suspect[0].startswith(f"GATE_PORT_SUSPECT {design.AAT[2]}: 0 of {episodes} episodes have a diagnosis accepted")
    assert "funnel diagnosis_accepted 0" in suspect[0] and lines[-1] == "GATE_MISS"
    other = tmp_path / "other"
    _gate(other)
    assert run_checks.main(["gate", str(other)]) == 0 and capsys.readouterr().out.splitlines()[-1] == "GATE_PASS"


def test_a_gate_whose_bar_no_count_reaches_or_lies_at_or_below_the_floor_is_refused(tmp_path, tmp_path_factory, monkeypatch, capsys):
    """Part one, section 5, with the bar of amendment 34.16: on starts that share one faulty set, the constant answer is right in
    every episode and no count of them separates from it, and with the floor of pass raised above four fifths in a copy of the
    file the bar of 4 of 5 lies below the floor; both times the check refuses the registration and reads nothing else."""
    _gate(tmp_path)
    scores = tmp_path / f"gate_{design.CD[0]}" / "scores.csv"
    with open(scores, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["block"] = _block()
    with open(scores, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert run_checks.main(["gate", str(tmp_path)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"GATE_REFUSED no count of the 5 episodes of {design.CD[2]} separates from the 5 of the constant answer by an exact "
                     "one-sided binomial test at 0.05, so the registration itself is refused"]
    other = tmp_path / "other"
    _gate(other)
    data = registered.load()
    data["descriptive"]["floor_of_pass"] = 0.9
    path = tmp_path_factory.mktemp("raised") / registered.PATH.name
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(registered, "PATH", path)
    assert run_checks.main(["gate", str(other)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        "GATE_REFUSED the bar 4 of 5 of the registered rule lies at or below the floor 0.900 of the starts the round played, so the "
        "registration itself is refused"]


def test_the_gate_refuses_runs_of_other_seeds_and_runs_of_the_test_set(tmp_path):
    _gate(tmp_path, seeds=[7, 8])
    with pytest.raises(SystemExit, match="play other seeds"):
        run_checks.main(["gate", str(tmp_path)])
    other = tmp_path / "other"
    _run(other, "cd", agent=design.CD[0], **{**GATE, "starts": "sampled"})
    _run(other, "pe", agent=design.PE[0], **GATE)
    with pytest.raises(SystemExit, match=f"0 finished development runs of {re.escape(design.AAT[2])}"):
        run_checks.main(["gate", str(other)])


def test_a_check_stops_on_an_empty_registered_number_and_names_it(tmp_path, tmp_path_factory, monkeypatch):
    _gate(tmp_path)
    _registered(tmp_path_factory, monkeypatch, valid_share=None)
    with pytest.raises(registered.OpenValue, match="development.valid_share"):
        run_checks.main(["gate", str(tmp_path)])
    _registered(tmp_path_factory, monkeypatch, max_harness_share=None)
    with pytest.raises(registered.OpenValue, match="development.max_harness_share"):
        run_checks.main(["round", str(tmp_path)])


def test_the_round_check_prints_the_longest_episode_against_half_the_wall_limit(tmp_path, capsys):
    """Every row of the main table on its preset gets one line with its unscored and harness counts, its longest episode against
    half the wall limit, its endpoint or context errors and its input tokens per model call; a row past a limit reads ROUND_MISS."""
    half = design.EPISODE_WALL_LIMIT_S / 2
    for row in design.AGENT_ROWS:
        slow = row == design.CD
        cells = {"wall_s": (str(half + 1 if slow else half), "10"), "llm_calls": ("2", "2"), "input_tokens": ("100", "300"),
                 "error": ("APIConnectionError: refused", "") if slow else ("", "BadRequestError: maximum context length is 32768 tokens")}
        _run(tmp_path, f"round_{row[0]}_{int(row[1])}", agent=row[0], cells=cells, **{**GATE, "diagnostics_server": row[1]})
    assert run_checks.main(["round", str(tmp_path)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert sum(1 for line in lines if line.startswith("ROUND ")) == len(design.AGENT_ROWS)
    ours = next(line for line in lines if line.startswith(f"ROUND {design.CD[2]}:"))
    assert f"longest episode {half + 1:.0f} s against {half:.0f} s" in ours and "endpoint or context errors 1" in ours and "per model call 100" in ours
    assert f"ROUND_MISS {design.CD[2]}: longest episode" in lines and lines[-1] == f"ROUND_MISS 1 of {len(design.AGENT_ROWS)} rows"


def test_a_run_of_a_requeued_or_suspended_job_and_a_run_that_slept_are_named(tmp_path, capsys):
    """Any suspension of the process is charged as thinking time, so the check names a run whose job
    was requeued (its ledger), whose job was suspended (the job record that the grid job writes beside its runs) or whose wall
    seconds before a tool call exceed the monotonic seconds, the trace of a sleep of the machine, with its longest wall gap.
    The monotonic seconds are read on the generation time clock alone, where the driver records them (0.0 in the other
    modes), and a call whose entry is null is left out and counted in the note."""
    quiet = ((1.0, 2.0), (1.0, 2.0))
    clocked = {"clock_mode": run_health_checks.WALL_CLOCK}
    _run(tmp_path, "clean", job={"slurm_job_id": "11"}, gaps=(quiet, quiet), **clocked)
    _run(tmp_path, "requeued", job={"slurm_job_id": "12", "slurm_restart_count": "1"}, gaps=(quiet, quiet), **clocked)
    _run(tmp_path, "suspended", job={"slurm_job_id": "13"}, gaps=(quiet, ((1.0, 400.0), (1.0, 400.0))), **clocked)
    _run(tmp_path, "slept", job={"slurm_job_id": "11"}, gaps=(quiet, ((1.0, 930.5), (1.0, 3.5))), **clocked)
    _run(tmp_path, "unclocked", job={"slurm_job_id": "11"}, gaps=(quiet, ((1.0, 930.5), (0.0, 0.0))), clock_mode="declared")
    _run(tmp_path, "partial", job={"slurm_job_id": "11"}, gaps=(quiet, ((1.0, 500.0), (1.0, None))), **clocked)
    (tmp_path / run_health_checks.JOB_RECORD).write_text("11|RUNNING|2026-01-01T00:00:00|Unknown|01:00:00|00:00:00|node001\n"
                                                  "13|RUNNING|2026-01-01T00:00:00|Unknown|01:00:00|1-00:06:40|node002\n", encoding="utf-8")
    assert run_checks.main(["suspect", str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        "SUSPECT requeued: job 12 restarted 1 time; longest wall gap before a tool call 2.0 s (seed000000)",
        "SUSPECT slept: 927.0 s of wall time beyond the monotonic clock before its tool calls; longest wall gap before a tool call 930.5 s (seed000001)",
        "SUSPECT suspended: job 13 suspended for 86800 s by its job record; longest wall gap before a tool call 400.0 s (seed000001)",
        "SUSPECT_NOTE 1 of 6 runs record no monotonic seconds for all or some of their tool calls, so a sleep of the machine cannot be read from those calls",
        "SUSPECT_RUNS 3 of 6 runs"]


def test_runs_without_the_monotonic_seconds_are_counted_and_a_clean_folder_passes(tmp_path, capsys):
    _run(tmp_path, "old", gaps=(((1.0,), None), ((2.0,), None)), clock_mode=run_health_checks.WALL_CLOCK)
    _run(tmp_path, "new", gaps=(((1.0,), (1.0,)), ((2.0,), (2.0,))), clock_mode=run_health_checks.WALL_CLOCK)
    assert run_checks.main(["suspect", str(tmp_path)]) == 0
    # the folder holds no job record, which the check names before the lines of the runs (test_a_grid_folder_without_a_job_record_is_named)
    assert capsys.readouterr().out.splitlines() == [
        f"SUSPECT_NOTE no job record beside the runs of {tmp_path}, so the suspended time of the job is unknown",
        "SUSPECT_NOTE 1 of 2 runs record no monotonic seconds for all or some of their tool calls, so a sleep of the machine cannot be read from those calls",
        "SUSPECT_RUNS 0 of 2 runs"]


def test_a_grid_folder_without_a_job_record_is_named(tmp_path, capsys):
    """A grid folder without the job record that the grid job writes beside its runs leaves the suspended time of the job unknown, so
    the check names the folder before the lines of the runs, and the exit status reads the runs alone."""
    runs = ["done"]
    for name in runs:
        _run(tmp_path, name, job={"slurm_job_id": "21"})
    assert run_checks.main(["suspect", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"SUSPECT_NOTE no job record beside the runs of {tmp_path}, so the suspended time of the job is unknown", f"SUSPECT_RUNS 0 of {len(runs)} runs"]


def test_a_job_record_that_the_limit_ended_is_read_and_named(tmp_path, capsys):
    """A job that the time limit ended leaves its lines in the state TIMEOUT, one per start of a requeued job: their suspended
    seconds are read as on any other line and name the run, and the state is named once for the folder."""
    runs = ["limited"]
    for name in runs:
        _run(tmp_path, name, job={"slurm_job_id": "21"}, gaps=(((1.0,), (1.0,)), ((2.0,), (2.0,))))
    (tmp_path / run_health_checks.JOB_RECORD).write_text("21|TIMEOUT|2026-01-01T00:00:00|2026-01-01T12:00:00|12:00:00|00:05:00|nodeA\n"
                                                  "21|TIMEOUT|2026-01-01T12:01:00|2026-01-02T00:01:00|12:00:00|00:05:00|nodeB\n", encoding="utf-8")
    assert run_checks.main(["suspect", str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        f"SUSPECT_NOTE the job record of {tmp_path} ends in the state TIMEOUT",
        "SUSPECT limited: job 21 suspended for 600 s by its job record; longest wall gap before a tool call 2.0 s (seed000001)",
        f"SUSPECT_RUNS {len(runs)} of {len(runs)} runs"]


def _timed(per_episode, episodes=2, runs_in_flight=4):
    """The run.json fields of a pilot run of episodes planned episodes whose elapsed seconds give per_episode seconds each."""
    start = datetime(2026, 1, 1)
    return {"started": start.isoformat(timespec="seconds"), "ended": (start + timedelta(seconds=per_episode * episodes)).isoformat(timespec="seconds"),
            "runs_in_flight": runs_in_flight}


def test_the_pilot_projects_a_registered_run_and_warns_with_the_jobs_it_needs_past_its_share_of_the_job_limit(tmp_path, capsys):
    """A registered run plays its episodes one after the other, so the pilot projects the run of each agent: the elapsed seconds per
    planned episode of its pilot run, at the runs in flight of the pilot, times the episodes of one registered run, against
    design.PROJECTION_SHARE of design.JOB_LIMIT_S, with the jobs the run needs at that share each. A projection past the share is a
    warning, PILOT_SLOW with the jobs of the slowest run, and the pilot passes, since the run continues in the next job; a
    projection under it prints no warning."""
    edge = float(design.PROJECTION_SHARE * design.JOB_LIMIT_S) / run_lines.STARTS
    slow, fits = math.ceil(edge) + 10, math.floor(edge) - 10
    agent, name = design.CD_MCP[0], design.CD_MCP[2]
    _pilot(tmp_path, **{row[0]: _timed(slow if row[0] == agent else fits) for row in design.TRAINING_ROWS})
    assert run_checks.main(["pilot", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    line = next(row for row in lines if row.startswith(f"PILOT projected run, {name}: "))
    assert line.split(": ", 1)[1].startswith(f"{slow} s per planned episode at runs in flight 4, times {run_lines.STARTS} episodes of a registered run")
    assert line.endswith(", 2 jobs") and sum(row.startswith("PILOT projected run, ") and row.endswith(", 1 job") for row in lines) == 2
    assert lines[-2].startswith(f"PILOT_SLOW {name}: the projected registered run passes 3/4 of the job limit at the runs in flight of the "
                                "pilot and needs 2 jobs") and lines[-1] == "PILOT_OK"
    fitting = tmp_path / "fits"
    _pilot(fitting, **{row[0]: _timed(fits) for row in design.TRAINING_ROWS})
    assert run_checks.main(["pilot", str(fitting)]) == 0
    out = capsys.readouterr().out
    assert "PILOT_SLOW" not in out and out.splitlines()[-1] == "PILOT_OK"


def test_a_pilot_run_taken_up_by_a_later_job_is_projected_from_the_wall_seconds_of_its_episodes(tmp_path, capsys):
    """The elapsed seconds of a pilot run that a later job took up again hold the wait between the jobs, so its projection reads the
    mean wall seconds of its episodes."""
    edge = float(design.PROJECTION_SHARE * design.JOB_LIMIT_S) / run_lines.STARTS
    agent, name = design.CD_MCP[0], design.CD_MCP[2]
    timing = {row[0]: _timed(math.floor(edge) - 10) for row in design.TRAINING_ROWS}
    _pilot(tmp_path, **{**timing, agent: {**_timed(10 * edge), "cells": {"wall_s": (60.0, 80.0)}}})
    folder = next(path for path in tmp_path.iterdir() if path.name.startswith(f"pilot_{agent}_"))
    (folder / ledger.LEDGER).write_text(json.dumps({"events": [{"tool": ledger.RESUME_TOOL}]}), encoding="utf-8")
    assert run_checks.main(["pilot", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"PILOT projected run, {name}: 70 s per planned episode" in out and "PILOT_SLOW" not in out


def test_the_pilot_refuses_only_an_episode_that_cannot_finish_inside_one_job(tmp_path, capsys, monkeypatch):
    """Every attempt of an episode ends at its wall limit, and the infrastructure may cost design.INFRASTRUCTURE_REPLAYS attempts
    more, so the pilot refuses, with PILOT_EPISODE_TOO_LONG, a wall limit whose attempts pass the job limit, and nothing else."""
    _pilot(tmp_path)
    monkeypatch.setattr(design, "EPISODE_WALL_LIMIT_S", design.JOB_LIMIT_S)
    assert run_checks.main(["pilot", str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == (
        f"PILOT_EPISODE_TOO_LONG {1 + design.INFRASTRUCTURE_REPLAYS} attempts of {design.JOB_LIMIT_S} s each pass the job limit of "
        f"{design.JOB_LIMIT_S} s, so an episode may never finish inside one job")


def test_the_job_limit_of_the_projection_is_the_time_limit_of_the_serving_job_and_of_the_grid_job():
    """design.JOB_LIMIT_S is the -t of CausalDiagnosis/cluster_jobs/2_serve_model/serve_model.sbatch, the job a registered run plays in, and of CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch."""
    for name in ("2_serve_model/serve_model.sbatch", "3_run/run_grid.sbatch"):
        limit = re.search(r"^#SBATCH -t (\d+):(\d\d):(\d\d)$", (ROOT / "CausalDiagnosis" / "cluster_jobs" / name).read_text(encoding="utf-8"), re.M)
        hours, minutes, seconds = (int(part) for part in limit.groups())
        assert hours * 3600 + minutes * 60 + seconds == design.JOB_LIMIT_S, name
