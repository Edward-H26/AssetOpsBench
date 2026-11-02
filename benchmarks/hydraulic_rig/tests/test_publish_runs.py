"""benchmarks/hydraulic_rig/publish_runs.py on a synthetic run folder: the published subset, the scrubbed endpoints, the
size cap on the model calls and the index."""
import json

from common import COLUMNS, write_rows
import publish_runs


def _run_folder(root, calls_size):
    run_dir = root / "static_plan_execute_m_20261101-000000"
    ep = run_dir / "seed000000"
    (ep / "hidden").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"label": run_dir.name, "agent": "plan_execute", "agent_label": "plan_execute_m", "model": "m",
                                                   "config_name": "static", "endpoint": "http://node:1234/v1", "judge_endpoint": "http://node:1235/v1",
                                                   "facts_path": "/some/where/hyd_facts.jsonl", "judge": "local"}), encoding="utf-8")
    row = {c: "" for c in COLUMNS}
    row.update({"seed": 0, "pass": True})
    write_rows(run_dir / "scores.csv", [row], COLUMNS)
    (run_dir / "summary.md").write_text("# Run\n", encoding="utf-8")
    (run_dir / "aggregate").mkdir()
    (run_dir / "aggregate" / "A1.csv").write_text("agent\n", encoding="utf-8")
    for name in ("report.md", "question.txt", "plan.txt", "answer.txt"):
        (ep / name).write_text(name, encoding="utf-8")
    (ep / "episode.json").write_text(json.dumps({"format": 2, "seed": 0, "facts_path": "/some/where/hyd_facts.jsonl"}), encoding="utf-8")
    (ep / "released.json").write_text(json.dumps({"format": 2, "events": []}), encoding="utf-8")
    (ep / "trajectory.json").write_text(json.dumps([{"step": 1, "server": "hydraulic_rig", "tool": "finish"}]), encoding="utf-8")
    (ep / "llm_calls.json").write_text(json.dumps([{"prompt": "x" * calls_size, "reply": "y"}]), encoding="utf-8")
    (ep / "hydraulic_rig_server.py").write_text("launcher", encoding="utf-8")
    (ep / "hidden" / "verdict.json").write_text(json.dumps({"verified": True, "pass": True}), encoding="utf-8")
    (ep / "hidden" / "oracle.json").write_text(json.dumps({"diagnosis": {"cooler": 100}}), encoding="utf-8")
    (ep / "hidden" / "log.json").write_text(json.dumps([{"event": "reset", "hidden": [100, 100, 0, 130]}]), encoding="utf-8")
    (ep / "hidden" / "score.json").write_text(json.dumps({"pass": True, "judge": {"base_url": "http://node:1235/v1", "pass": None}}), encoding="utf-8")
    return run_dir


def test_published_subset_holds_the_listed_files_and_no_endpoint(tmp_path):
    run_dir = _run_folder(tmp_path / "runs", calls_size=10)
    published = publish_runs.publish(tmp_path / "runs", tmp_path / "out", llm_cap=1000)
    assert published == {run_dir.name: 1}
    out = tmp_path / "out" / run_dir.name
    assert sorted(p.name for p in out.iterdir()) == ["aggregate", "run.json", "scores.csv", "seed000000", "summary.md"]
    ep = out / "seed000000"
    assert sorted(p.name for p in ep.iterdir()) == ["answer.txt", "hidden", "llm_calls.json", "plan.txt", "question.txt", "released.json", "report.md", "trajectory.json"]
    assert sorted(p.name for p in (ep / "hidden").iterdir()) == ["oracle.json", "score.json", "verdict.json"]
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert meta["endpoint"] is None and meta["judge_endpoint"] is None and meta["facts_path"] == "hyd_facts.jsonl" and meta["model"] == "m"
    score = json.loads((ep / "hidden" / "score.json").read_text(encoding="utf-8"))
    assert score["judge"]["base_url"] is None and score["judge"]["pass"] is None
    assert "node:1234" not in "".join(p.read_text(encoding="utf-8") for p in out.rglob("*") if p.is_file())
    index = (tmp_path / "out" / "INDEX.md").read_text(encoding="utf-8")
    assert f"| [{run_dir.name}]({run_dir.name}/summary.md) | 1 | plan_execute_m | m | static |" in index


def test_model_calls_above_the_cap_are_replaced_by_a_note(tmp_path):
    run_dir = _run_folder(tmp_path / "runs", calls_size=5000)
    publish_runs.publish(tmp_path / "runs", tmp_path / "out", llm_cap=1000)
    ep = tmp_path / "out" / run_dir.name / "seed000000"
    assert not (ep / "llm_calls.json").exists()
    note = (ep / "llm_calls.omitted.txt").read_text(encoding="utf-8")
    assert "above the published cap of 1000 bytes" in note


def test_scrubbed_sets_endpoints_to_null_at_any_depth():
    data = {"endpoint": "http://a", "nested": [{"base_url": "http://b", "keep": 1}], "facts_path": "/x/y/z.jsonl", "other": "v"}
    assert publish_runs.scrubbed(data) == {"endpoint": None, "nested": [{"base_url": None, "keep": 1}], "facts_path": "z.jsonl", "other": "v"}
