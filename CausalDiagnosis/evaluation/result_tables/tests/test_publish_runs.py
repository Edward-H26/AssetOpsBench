"""CausalDiagnosis/evaluation/result_tables/publish_runs.py on a synthetic run folder: the published subset copied byte for byte, the size cap on
the model calls and the index. The drivers write a run folder in its public form (public_record.py), so nothing is scrubbed."""
import json
import shutil

from CausalDiagnosis.evaluation.result_tables import publish_runs
from CausalDiagnosis.evaluation.result_tables import pooling_rules
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS
from CausalDiagnosis.utils.csv_rows import write_rows


def _run_folder(root, calls_size):
    run_dir = root / "static_plan_execute_m_20261101-000000"
    ep = run_dir / "seed000000"
    (ep / "hidden").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"label": run_dir.name, "agent": "plan_execute", "agent_label": "plan_execute_m", "model": "m",
                                                   "config_name": "static", "endpoint": None, "judge_endpoint": None,
                                                   "facts_path": "hyd_facts.jsonl", "judge": "local"}), encoding="utf-8")
    row = {c: "" for c in COLUMNS}
    row.update({"seed": 0, "pass": True})
    write_rows(run_dir / "scores.csv", [row], COLUMNS)
    (run_dir / "summary.md").write_text("# Run\n", encoding="utf-8")
    (run_dir / "aggregate").mkdir()
    (run_dir / "aggregate" / "A1.csv").write_text("agent\n", encoding="utf-8")
    for name in ("report.md", "question.txt", "plan.txt", "answer.txt"):
        (ep / name).write_text(name, encoding="utf-8")
    (ep / "episode.json").write_text(json.dumps({"format": 2, "seed": 0, "facts_path": "hyd_facts.jsonl"}), encoding="utf-8")
    (ep / "released.json").write_text(json.dumps({"format": 3, "events": []}), encoding="utf-8")
    (ep / "trajectory.json").write_text(json.dumps([{"step": 1, "server": "hydraulic_rig", "tool": "finish"}]), encoding="utf-8")
    (ep / "llm_calls.json").write_text(json.dumps([{"prompt": "x" * calls_size, "reply": "y"}]), encoding="utf-8")
    (ep / "hydraulic_rig_server.py").write_text("launcher", encoding="utf-8")
    (ep / "hidden" / "verdict.json").write_text(json.dumps({"verified": True, "pass": True}), encoding="utf-8")
    (ep / "hidden" / "oracle.json").write_text(json.dumps({"diagnosis": {"cooler": 100}}), encoding="utf-8")
    (ep / "hidden" / "log.json").write_text(json.dumps([{"event": "reset", "hidden": [100, 100, 0, 130]}]), encoding="utf-8")
    (ep / "hidden" / "score.json").write_text(json.dumps({"pass": True, "judge": {"base_url": None, "pass": None}}), encoding="utf-8")
    return run_dir


def test_published_subset_holds_the_listed_files_byte_for_byte(tmp_path):
    run_dir = _run_folder(tmp_path / "runs", calls_size=10)
    published = publish_runs.publish(tmp_path / "runs", tmp_path / "out", llm_cap=1000)
    assert published == {run_dir.name: 1}
    out = tmp_path / "out" / run_dir.name
    assert sorted(p.name for p in out.iterdir()) == ["aggregate", "run.json", "scores.csv", "seed000000", "summary.md"]
    ep = out / "seed000000"
    assert sorted(p.name for p in ep.iterdir()) == ["answer.txt", "hidden", "llm_calls.json", "plan.txt", "question.txt", "released.json", "report.md", "trajectory.json"]
    assert sorted(p.name for p in (ep / "hidden").iterdir()) == ["oracle.json", "score.json", "verdict.json"]
    assert all((run_dir / p.relative_to(out)).read_bytes() == p.read_bytes() for p in out.rglob("*") if p.is_file())
    index = (tmp_path / "out" / "INDEX.md").read_text(encoding="utf-8")
    assert f"| [{run_dir.name}]({run_dir.name}/summary.md) | 1 | plan_execute_m | m | static |" in index


def test_two_runs_of_one_label_are_published_apart_and_keep_their_training_variant(tmp_path):
    """A trained agent and the agent without training carry one agent label, so the published run.json is where a reader of the copy
    sees which variant a run played."""
    released = _run_folder(tmp_path / "runs", calls_size=10)
    trained = released.with_name("static_plan_execute_m_20261101-000500")
    shutil.copytree(released, trained)
    variants = ((released, "as_released"), (trained, "gepa"))
    for run_dir, variant in variants:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        (run_dir / "run.json").write_text(json.dumps({**meta, "training_variant": variant}), encoding="utf-8")
    assert publish_runs.publish(tmp_path / "runs", tmp_path / "out", llm_cap=1000) == {released.name: 1, trained.name: 1}
    for run_dir, variant in variants:
        assert json.loads((tmp_path / "out" / run_dir.name / "run.json").read_text(encoding="utf-8"))["training_variant"] == variant


def test_model_calls_above_the_cap_are_replaced_by_a_note(tmp_path):
    run_dir = _run_folder(tmp_path / "runs", calls_size=5000)
    publish_runs.publish(tmp_path / "runs", tmp_path / "out", llm_cap=1000)
    ep = tmp_path / "out" / run_dir.name / "seed000000"
    assert not (ep / "llm_calls.json").exists()
    note = (ep / "llm_calls.omitted.txt").read_text(encoding="utf-8")
    assert "above the published cap of 1000 bytes" in note


def test_a_run_of_the_online_loop_publishes_its_result_and_candidates_without_the_reflection_calls(tmp_path):
    """A run of gepa_prompt_optimization/ publishes gepa_online.json and the texts of every candidate under candidates/ byte for byte,
    which TRAINING.csv names in checkpoint, while the calls of the reflection model under reflection/ stay in the synced run folder."""
    run_dir = _run_folder(tmp_path / "runs", calls_size=10)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    (run_dir / "run.json").write_text(json.dumps({**meta, "mode": "gepa_online", "gepa": {"card": "verifier", "reflection_endpoint": None}}), encoding="utf-8")
    (run_dir / "gepa_online.json").write_text(json.dumps({"status": "finished", "reflection_endpoint": None}), encoding="utf-8")
    (run_dir / "candidates").mkdir()
    (run_dir / "candidates" / "c000_system_prompt.txt").write_text("You plan.", encoding="utf-8")
    (run_dir / "candidates" / "c000_threshold_card.json").write_text('{"reserve": 1}\n', encoding="utf-8")
    (run_dir / "reflection").mkdir()
    (run_dir / "reflection" / "c001_attempt1.json").write_text(json.dumps({"prompt": "p", "reply": "r"}), encoding="utf-8")
    publish_runs.publish(tmp_path / "runs", tmp_path / "out", llm_cap=1000)
    out = tmp_path / "out" / run_dir.name
    assert sorted(p.name for p in out.iterdir()) == ["aggregate", "candidates", "gepa_online.json", "run.json", "scores.csv", "seed000000", "summary.md"]
    assert all((run_dir / p.relative_to(out)).read_bytes() == p.read_bytes() for p in out.rglob("*") if p.is_file())


def test_run_folders_that_no_table_reads_stay_unpublished_and_the_index_names_them(tmp_path):
    """publish_runs.py takes the refusals of results_index, so a run on the memory
    backend and a run of the retired mode gepa are not published and INDEX.md names each with its reason."""
    run_dir = _run_folder(tmp_path / "runs", calls_size=10)
    for name, extra in (("memory_run", {"backend": {"kind": "memory"}}), ("gepa_run", {"mode": "gepa"})):
        copy = shutil.copytree(run_dir, tmp_path / "runs" / name)
        meta = json.loads((copy / "run.json").read_text(encoding="utf-8"))
        (copy / "run.json").write_text(json.dumps({**meta, "label": name, **extra}), encoding="utf-8")
    assert publish_runs.publish(tmp_path / "runs", tmp_path / "out", llm_cap=1000) == {run_dir.name: 1}
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["INDEX.md", run_dir.name]
    index = (tmp_path / "out" / "INDEX.md").read_text(encoding="utf-8")
    assert f"| gepa_run | {pooling_rules.RETIRED_GEPA_MODE} |" in index and f"| memory_run | {pooling_rules.MEMORY_BACKEND} |" in index
