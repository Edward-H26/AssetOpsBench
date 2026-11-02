"""The audit report of an episode built from synthetic public files, the one line tool summaries, and the run summary;
none of them reads a hidden file, and the report never shows a hidden value."""
import json

from common import COLUMNS, write_rows
from report import episode_report, run_summary, summarize_response

HIDDEN_COMBINATION = [3, 100, 2, 90]


def _envelope(result, **status):
    base = {"time": 4, "day": None, "budget_left": 290, "steps_left": 55, "open_work_orders": [], "episode_over": False}
    base.update(status)
    return json.dumps({**base, "result": result})


def _episode_folder(tmp_path):
    ep = tmp_path / "seed000007"
    ep.mkdir()
    (ep / "episode.json").write_text(json.dumps({"format": 2, "seed": 7, "actions": [], "think_cycles": [], "tool_calls": [
        {"tool": "read_sensors", "arguments": {"sensor_group": "tier1", "cycles": 2},
         "response": _envelope({"sensor_group": "tier1", "served": [{"n": 1}, {"n": 2}], "repeated": False})},
        {"tool": "create_work_order", "arguments": {"component": "pump", "prediction_id": "", "evidence_ids": ""},
         "response": json.dumps({"refused": "a diagnosis or a declaration precedes a work order", "time": 2, "budget_left": 298, "steps_left": 58,
                                 "open_work_orders": [], "episode_over": False})}]}), encoding="utf-8")
    (ep / "released.json").write_text(json.dumps({
        "format": 2, "asset_id": "hyd_1", "task_text": "Asset hyd_1 was flagged.", "guidance_level": 2,
        "status": {"time": 9, "budget_left": 240, "steps_left": 50, "open_work_orders": [], "episode_over": True},
        "events": [{"event": "read", "t_start": 0, "t_end": 2, "class": "read"},
                   {"event": "refused", "action": {"type": "create_work_order", "component": "pump"}, "reason": "a diagnosis or a declaration precedes a work order", "class": "write"},
                   {"event": "work_order_created", "id": "wo-1", "component": "pump", "class": "write"},
                   {"event": "work_order_closed", "id": "wo-1", "component": "pump", "source": "agent", "class": "system"},
                   {"event": "finish", "summary": "The pump was replaced and the rig recovered.", "class": "write"},
                   {"event": "end", "reason": "finish", "time": 9, "cost": 60, "downtime_total": 4, "class": "system"}],
        "observations": []}), encoding="utf-8")
    (ep / "predictions.json").write_text(json.dumps([{"id": "pred/1", "action": "replace:pump", "recovery": 0.5}]), encoding="utf-8")
    (ep / "llm_calls.json").write_text(json.dumps([{"prompt": "p", "reply": "r", "input_tokens": 10, "output_tokens": 2, "seconds": 0.5}]), encoding="utf-8")
    (ep / "plan.txt").write_text("#Task1: read\n#Server1: hydraulic_rig\n#Tool1: read_sensors\n", encoding="utf-8")
    (ep / "trajectory.json").write_text(json.dumps([
        {"step": 1, "task": "read", "server": "hydraulic_rig", "tool": "read_sensors", "tool_args": {"sensor_group": "tier1", "cycles": 2},
         "response": _envelope({"sensor_group": "tier1", "served": [{"n": 1}, {"n": 2}], "repeated": False}), "error": None},
        {"step": 2, "task": "rank", "server": "causal_diagnosis", "tool": "rank_root_causes", "tool_args": {"top_k": 2},
         "response": json.dumps({"message": "two candidates ranked", "candidates": []}), "error": None},
        {"step": 3, "task": "broken", "server": "hydraulic_rig", "tool": "finish", "tool_args": {}, "response": "", "error": "Unknown tool"}]), encoding="utf-8")
    hidden = ep / "hidden"
    hidden.mkdir()
    (hidden / "oracle.json").write_text(json.dumps({"diagnosis": dict(zip(("cooler", "valve", "pump", "accumulator"), HIDDEN_COMBINATION))}), encoding="utf-8")
    (hidden / "verdict.json").write_text(json.dumps({"pass": True, "failure_class": "none"}), encoding="utf-8")
    return ep


def test_episode_report_shows_the_public_record_and_no_hidden_value(tmp_path):
    ep = _episode_folder(tmp_path)
    meta = {"config_name": "static", "agent": "plan_execute", "model": "m", "servers": ["hydraulic_rig", "causal_diagnosis"], "clock_mode": "declared",
            "pilot": {"assumptions": ["A1 (mapping): one cycle per day", "A2 (dates): four of ten"], "date0": "2022-07-01",
                      "anchors": {"seed000007": {"date": "2023-01-28", "cycle_index": 211, "stable": False, "anchored_cycle_index": 210, "block_offset": 3}}}}
    report = episode_report(ep, meta)
    assert "Config static, agent plan_execute (m), servers hydraulic_rig, causal_diagnosis, seed 7, clock mode declared, guidance level 2." in report
    assert "Task text: Asset hyd_1 was flagged." in report
    assert "- A1 (mapping): one cycle per day" in report and "anchored at 2023-01-28 (cycle 211 from 2022-07-01, an unstable cycle, so the nearest stable cycle 210" in report
    assert "| 1 | hydraulic_rig.read_sensors | `{\"sensor_group\":\"tier1\",\"cycles\":2}` | served 2 cycle(s) of tier1; time 4, budget left 290, steps left 55, open work orders 0, over False |" in report
    assert "| 2 | causal_diagnosis.rank_root_causes | `{\"top_k\":2}` | two candidates ranked |" in report
    assert "| 3 | hydraulic_rig.finish | `{}` | error: Unknown tool |" in report
    assert "| pred/1 | replace:pump | recovery |" in report
    assert "1 work order(s) created, 1 closed; 1 refused call(s)." in report and "End reason 'finish' at cycle 9, cost 60, downtime 4." in report
    assert "> The pump was replaced and the rig recovered." in report
    assert "1 calls, 10 input tokens, 2 output tokens, 0.5 s in the model." in report
    assert "are under hidden/ and are not shown here" in report
    assert json.dumps(HIDDEN_COMBINATION) not in report and "pass" not in report.lower().replace("passes", "") and "verdict" not in report.replace("the verdict and the scores", "")


def test_episode_report_without_a_trajectory_reads_the_servers_own_record(tmp_path):
    ep = _episode_folder(tmp_path)
    (ep / "trajectory.json").unlink()
    (ep / "plan.txt").unlink()
    report = episode_report(ep, {"config_name": "static", "agent": "oracle", "servers": ["hydraulic_rig"]})
    assert "## Plan" not in report
    assert "| 1 | hydraulic_rig.read_sensors |" in report
    assert "| 2 | hydraulic_rig.create_work_order | `{\"component\":\"pump\",\"prediction_id\":\"\",\"evidence_ids\":\"\"}` | refused: a diagnosis or a declaration precedes a work order; time 2" in report


def test_summarize_response_reads_envelopes_messages_and_errors():
    assert summarize_response("hydraulic_rig", "advance_time", _envelope({"time": 9})) == "advanced to cycle 9; time 4, budget left 290, steps left 55, open work orders 0, over False"
    assert summarize_response("hydraulic_rig", "summarize_changes_since", _envelope({"since": 0, "alarms": [{"channel": "x", "first_t": 3}], "work_orders_closed": []})).startswith("1 alarm(s) and 0 work order(s) closed since cycle 0;")
    assert summarize_response("hydraulic_rig", "create_work_order", _envelope({"work_order": {"id": "wo-1", "component": "pump", "expected_close_t": 9}})).startswith("work order wo-1 on pump, expected to close at cycle 9;")
    assert summarize_response("hydraulic_rig", "submit_diagnosis", _envelope({"diagnosis": {"pump": 1}})).startswith("diagnosis {'pump': 1};")
    assert summarize_response("hydraulic_rig", "declare_insufficient_evidence", _envelope({"separating_test": {"sensor_group": "tier3", "cycles": 1}})).startswith("declared insufficient evidence, separating test")
    assert summarize_response("hydraulic_rig", "get_asset_profile", _envelope({"components": ["a", "b"]})).startswith("profile of 2 components;")
    assert summarize_response("hydraulic_rig", "finish", _envelope({"ended": True}, episode_over=True)).endswith("over True")
    assert summarize_response("hydraulic_rig", "get_failure_modes", _envelope([{"code": "x"}])).startswith("[{'code': 'x'}];")
    assert summarize_response("causal_diagnosis", "verify_outcome", json.dumps({"error": "unknown prediction_id"})) == "error: unknown prediction_id"
    assert summarize_response("causal_diagnosis", "verify_outcome", "", "boom") == "error: boom"
    assert summarize_response("hydraulic_rig", "read_sensors", "not json") == "not json"


def test_run_summary_counts_episodes_and_files_and_states_no_score(tmp_path):
    ep = _episode_folder(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps({"config_name": "static", "agent": "plan_execute", "model": "m", "servers": ["hydraulic_rig"], "clock_mode": "declared",
                                                   "guidance_level": 2, "prices": "placeholder", "verifier_scope": "writes", "judge": "none", "episodes_planned": 1,
                                                   "started": "t0", "ended": "t1", "pilot": {"assumptions": ["A1 (mapping): one cycle per day"]}}), encoding="utf-8")
    row = {c: "" for c in COLUMNS}
    row.update({"seed": 7, "scenario_id": "", "pass": True, "cost": 60, "error": ""})
    write_rows(tmp_path / "scores.csv", [row], COLUMNS)
    summary = run_summary(tmp_path)
    assert "Episodes planned 1, episode folders 1, rows in scores.csv 1, rows with an error 0." in summary
    assert "| episode.json | 1 |" in summary and "| hidden/oracle.json | 1 |" in summary and "| hidden/log.json | 0 |" in summary
    assert "- A1 (mapping): one cycle per day" in summary
    assert "[seed000007](seed000007/report.md)" in summary
    assert "60" not in summary.replace("t0", "").replace("t1", "") and "True" not in summary
    assert ep.exists()
