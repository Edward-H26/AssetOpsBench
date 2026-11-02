"""The driver through the two MCP servers: the scripted plan reaches every tool of both servers, the reference policies
play through the action server with the served log equal to the shadow's, the model agent and the static agent run
through the real OpenAI compatible client against a local stand-in endpoint, the pilot anchors every scenario, and the
options land in run.json. Every episode is verified from its log; the tests assert relations, never a recorded value.
Run folders go under pytest's tmp_path, outside the repository."""
import csv
import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from common import CAUSAL_TOOLS, COLUMNS, HIDDEN, HIDDEN_FILES, RIG_TOOLS, as_bool, read_json
from envs.hydraulic_rig.data import COMPONENTS, build_blocks, default_facts_path, load_cycles
from pilot import anchors

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)
if importlib.util.find_spec("envs.hydraulic_rig.verifier") is None or not hasattr(__import__("envs.hydraulic_rig.env", fromlist=["x"]), "HydraulicRigEnv"):
    pytest.skip("waiting for env-core's HydraulicRigEnv and the verifier", allow_module_level=True)

import run_episodes  # noqa: E402


def _rows(run_dir):
    with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run(args, out):
    """One run into a fresh folder under tmp_path; returns the run folder, the rows of scores.csv and run.json."""
    run_episodes.main([*args, "--out", str(out)])
    runs = [p for p in Path(out).iterdir() if p.is_dir()]
    assert len(runs) == 1
    return runs[0], _rows(runs[0]), read_json(runs[0] / "run.json")


def test_scripted_plan_reaches_every_tool_of_both_servers_and_writes_every_file(tmp_path):
    run_dir, rows, meta = _run(["--config", "static", "--agent", "scripted", "--seeds", "1"], tmp_path / "scripted")
    assert len(rows) == 1 and rows[0]["agent"] == "scripted" and rows[0]["error"] == "", rows[0]["error"]
    assert list(rows[0]) == list(COLUMNS)
    ep_dir = run_dir / "seed000000"
    for name in ("episode.json", "released.json", "predictions.json", "trajectory.json", "llm_calls.json", "plan.txt", "answer.txt", "report.md", "question.txt"):
        assert (ep_dir / name).exists(), name
    for name in HIDDEN_FILES:
        assert (ep_dir / HIDDEN / name).exists(), name
    trajectory = json.loads((ep_dir / "trajectory.json").read_text(encoding="utf-8"))
    assert len(trajectory) == len(run_episodes.ScriptedBackend.STEPS) and all(t["error"] is None for t in trajectory)
    assert {(t["server"], t["tool"]) for t in trajectory} == {("hydraulic_rig", n) for n in RIG_TOOLS} | {("causal_diagnosis", n) for n in CAUSAL_TOOLS}
    predictions = json.loads((ep_dir / "predictions.json").read_text(encoding="utf-8"))
    assert [p["id"] for p in predictions] == ["pred/1"]
    # the declaration after the diagnosis is the one refused call of the plan
    assert rows[0]["refusals"] == "1" and rows[0]["tool_calls"] == str(sum(1 for t in trajectory if t["server"] == "hydraulic_rig"))
    verdict = read_json(ep_dir / HIDDEN / "verdict.json")
    assert verdict["verified"] and rows[0]["pass"] == str(verdict["pass"]) and rows[0]["end_reason"] == "finish"
    assert rows[0]["cost_oracle"] != "" and rows[0]["reward"] != ""
    # static starts lie in the diagnostics server's grade table, so the grade was looked up and passed as the tri-state
    score = read_json(ep_dir / HIDDEN / "score.json")
    assert score["grade_source"] == "table" and meta["grade_lookup"] is True
    assert verdict["grade"] is None or set(verdict["grade"]) == {"separable", "cost"}
    # the public files hold no hidden field and the report no hidden value
    released = (ep_dir / "released.json").read_text(encoding="utf-8")
    assert not any(f'"{k}"' in released for k in ("hidden", "combination_after", "unnecessary", "seed", "schedule", "final"))
    report = (ep_dir / "report.md").read_text(encoding="utf-8")
    oracle = read_json(ep_dir / HIDDEN / "oracle.json")
    assert "## Tool calls" in report and json.dumps([oracle["diagnosis"][c] for c in COMPONENTS]) not in report
    assert "verdict" not in report.replace("the verdict and the scores", "")
    assert meta["servers"] == ["hydraulic_rig", "causal_diagnosis"] and meta["briefing"] == run_episodes.BRIEFING_VERSION and meta["mode"] == "gym"
    assert meta["config"]["clock_mode"] == "declared" and meta["episodes_planned"] == 1 and meta["pilot"] is None
    assert meta["abstention_rule"] == "decision" and meta["abstention_cut"] is None
    # the recorded ranking model is the one the diagnostics server reports in its ranking answer, and both are the server's default
    ranked = next(t for t in trajectory if t["tool"] == "rank_root_causes")
    reported = json.loads(ranked["response"])["ranking_model"]
    assert meta["ranking_model"] == rows[0]["ranking_model"] == reported == run_episodes.DEFAULT_RANKING_MODEL
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Episodes planned 1, episode folders 1, rows in scores.csv 1" in summary and "| hidden/verdict.json | 1 |" in summary


def test_oracle_and_do_nothing_play_through_the_server_and_the_oracle_outranks_do_nothing(tmp_path):
    runs = {}
    for agent in ("oracle", "do_nothing"):
        run_dir, rows, meta = _run(["--config", "static", "--agent", agent, "--seeds", "3", "--first-seed", "0", "--no-diagnostics"], tmp_path / agent)
        assert meta["servers"] == ["hydraulic_rig"] and meta["config"]["diagnostics_server"] is False
        assert [r["seed"] for r in rows] == ["0", "1", "2"] and all(r["error"] == "" for r in rows), [r["error"] for r in rows]
        assert all(as_bool(r["replay_matches"]) for r in rows)
        runs[agent] = {r["seed"]: r for r in rows}
        for r in rows:
            assert (run_dir / f"seed{int(r['seed']):06d}" / HIDDEN / "log.json").exists()
    for seed in ("0", "1", "2"):
        oracle, nothing = runs["oracle"][seed], runs["do_nothing"][seed]
        assert as_bool(oracle["recovered"]) and as_bool(oracle["success"])
        assert not as_bool(nothing["pass"]) and not as_bool(nothing["success"]) and nothing["failure_class"] != ""
        assert oracle["block"] == nothing["block"] and oracle["separable"] == nothing["separable"]  # the same seed, the same hidden start
        # the verifier's success follows the grade: on a separable start the oracle diagnoses and passes; on an inseparable
        # start it declares instead, which is the correct abstention; either way it is the success the reward follows (RFC 008)
        if as_bool(oracle["separable"]):
            assert as_bool(oracle["pass"]) and as_bool(oracle["diagnosis_all_correct"]) and not as_bool(oracle["abstained"])
        else:
            assert as_bool(oracle["abstained"]) and as_bool(oracle["abstention_correct"]) and not as_bool(oracle["pass"])
        assert float(oracle["reward"]) > float(nothing["reward"])


STATIC_REPLY = "The pump is faulty and should be replaced; the cooler is healthy."


def _inseparable_seed(tmp_path, limit=60):
    """The first seed of the static preset whose start the grade table marks inseparable, or None within the limit."""
    from common import load_config, load_world
    from envs.hydraulic_rig.env import HydraulicRigEnv
    from servers.hydraulic_rig import episode

    facts_path, cycles, blocks = load_world()
    _, config = load_config("static")
    tmp_path.mkdir(parents=True, exist_ok=True)
    episode.start(tmp_path / "episode.json", 0, config, facts_path)
    released = read_json(tmp_path / "released.json")
    env = HydraulicRigEnv(cycles, blocks, config)
    for seed in range(limit):
        env.reset(seed)
        grade, source = run_episodes.grade_of_start(env.oracle(), released, run_episodes.DEFAULT_RANKING_MODEL)  # the table the runs use
        if source == "table" and grade is not None and not grade["separable"]:
            return seed
    return None


def test_the_oracle_declares_on_an_inseparable_start_and_outranks_do_nothing(tmp_path):
    seed = _inseparable_seed(tmp_path / "probe")
    if seed is None:
        pytest.skip("no inseparable start among the first seeds of the static preset under the placeholder grade table")
    rows = {}
    for agent in ("oracle", "do_nothing"):
        _run_dir, agent_rows, _meta = _run(["--config", "static", "--agent", agent, "--seeds", "1", "--first-seed", str(seed), "--no-diagnostics"], tmp_path / agent)
        assert agent_rows[0]["error"] == "" and as_bool(agent_rows[0]["replay_matches"])
        rows[agent] = agent_rows[0]
    oracle, nothing = rows["oracle"], rows["do_nothing"]
    assert oracle["separable"] == "False" and nothing["separable"] == "False"
    # the oracle declares instead of diagnosing, repairs every faulty component, and is a success under the grade
    assert as_bool(oracle["abstained"]) and as_bool(oracle["abstention_correct"]) and as_bool(oracle["success"]) and not as_bool(oracle["pass"])
    assert as_bool(oracle["recovered"]) and not as_bool(oracle["commitment_on_inseparable"])
    assert not as_bool(nothing["success"]) and float(oracle["reward"]) > float(nothing["reward"])


def _stand_in_endpoint(fixed_reply=None):
    """A local OpenAI compatible endpoint that answers every chat completion with the next reply of the scripted plan
    (or with one fixed reply), so the model agents run through the real client, request and response path without a
    model."""
    script = run_episodes.ScriptedBackend()
    script.start_episode()
    requests = []

    def next_reply():
        if fixed_reply is not None:
            return fixed_reply
        try:
            return script.generate("")
        except StopIteration:
            return STATIC_REPLY

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - the http.server hook name
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            reply = {"id": f"stand-in-{len(requests)}", "object": "chat.completion", "created": int(time.time()), "model": body["model"],
                     "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": next_reply()}}],
                     "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


def test_model_agent_runs_through_the_openai_compatible_client_with_a_judge(tmp_path, monkeypatch):
    server, requests = _stand_in_endpoint()
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        with pytest.raises(SystemExit, match="self judging"):
            run_episodes.main(["--config", "static", "--agent", "plan_execute", "--model", "vendor/stand-in-1.0", "--base-url", base, "--judge", "local",
                               "--out", str(tmp_path / "refused")])
        run_dir, rows, meta = _run(["--config", "static", "--agent", "plan_execute", "--model", "vendor/stand-in-1.0", "--base-url", base, "--seeds", "1",
                                    "--judge", "local", "--judge-model", "vendor/stand-in-judge"], tmp_path / "model")
        assert "." not in run_dir.name and "_vendor_stand-in-1_0_" in run_dir.name
        assert len(rows) == 1 and rows[0]["error"] == "" and rows[0]["agent"] == "plan_execute_vendor/stand-in-1.0" and rows[0]["model"] == "vendor/stand-in-1.0"
        steps = len(run_episodes.ScriptedBackend.STEPS)
        # the agent's calls are the plan, one per step and the summary; the judge's one call goes to the same stand in
        assert rows[0]["llm_calls"] == str(steps + 2) and len(requests) == steps + 3 and rows[0]["input_tokens"] == str(11 * (steps + 2))
        assert all(path == "/v1/chat/completions" for path, _ in requests)
        assert [body["model"] for _, body in requests].count("vendor/stand-in-judge") == 1
        assert meta["endpoint"] == base and meta["model_id"] == "tokenrouter/vendor/stand-in-1.0" and meta["model"] == "vendor/stand-in-1.0"
        assert meta["judge_model"] == "vendor/stand-in-judge" and meta["judge_endpoint"] == base
        assert rows[0]["causal_calls"] == str(len(CAUSAL_TOOLS))
        # the stand in answers the judge with prose, not a rubric, so the judge record exists and its pass is empty
        judged = read_json(run_dir / "seed000000" / HIDDEN / "judge.json")
        assert judged["protocol"] == "single" and judged["pass"] is None and rows[0]["judge_pass"] == ""
    finally:
        server.shutdown()


def test_static_agent_answers_without_the_gym_and_is_judged_and_parsed(tmp_path, monkeypatch):
    server, requests = _stand_in_endpoint(fixed_reply=STATIC_REPLY)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        # every reply of the stand in is one prose sentence, the static answer of both episodes and the judge's reply,
        # which is no rubric, so the judge records no pass
        run_dir, rows, meta = _run(["--config", "static", "--agent", "static", "--model", "vendor/stand-in-1.0", "--base-url", base, "--seeds", "2",
                                    "--no-grade", "--judge", "local", "--judge-model", "vendor/stand-in-judge"], tmp_path / "static")
        assert meta["mode"] == "static" and meta["agent"] == "static" and len(rows) == 2 and all(r["error"] == "" for r in rows), [r["error"] for r in rows]
        for row in rows:
            ep_dir = run_dir / f"seed{int(row['seed']):06d}"
            assert row["end_reason"] == "static" and row["tool_calls"] == "0" and row["pass"] == "" and row["llm_calls"] == "1"
            assert row["static_faulty_set_correct"] in ("True", "False") and row["judge_pass"] == ""
            for name in ("prompt.txt", "answer.txt", "question.txt", "llm_calls.json", "report.md", "released.json", "episode.json"):
                assert (ep_dir / name).exists(), name
            for name in ("oracle.json", "log.json", "static.json", "judge.json", "score.json"):
                assert (ep_dir / HIDDEN / name).exists(), name
            static = read_json(ep_dir / HIDDEN / "static.json")
            oracle = read_json(ep_dir / HIDDEN / "oracle.json")
            assert static["features_shown"] > 0 and static["failure_modes_shown"] > 0
            assert static["truth_faulty"] == [c for c in COMPONENTS if oracle["diagnosis"][c] != run_episodes.HEALTHY[c]]
            prompt = (ep_dir / "prompt.txt").read_text(encoding="utf-8")
            assert "You cannot call any tool" in prompt and "Failure modes of the asset" in prompt and "Pressure_PS1_mean" in prompt
            # the facts shown are features only: no label of the facts file and no hidden level reaches the prompt
            assert "labels" not in prompt and "failure_profile" not in prompt and json.dumps(oracle["diagnosis"]) not in prompt
            report = (ep_dir / "report.md").read_text(encoding="utf-8")
            assert "## Static mode" in report and "which the gym charges for" in report and "## Static answer" in report
            assert "trajectory.json" not in [p.name for p in ep_dir.iterdir()]
            parsed = read_json(ep_dir / HIDDEN / "static.json")["parsed"]
            assert parsed["faulty"] == ["pump"] and parsed["repaired"] == ["pump"] and parsed["sentences"]["cooler"] == []
            judge = read_json(ep_dir / HIDDEN / "judge.json")
            assert judge["mode"] == "static" and judge["answer"] == STATIC_REPLY and judge["pass"] is None
        # one model call per episode and one judge call per episode reached the stand in
        assert len(requests) == 4
    finally:
        server.shutdown()


def test_pilot_anchors_every_scenario_at_its_date_and_prints_the_assumptions(tmp_path):
    run_dir, rows, meta = _run(["--config", "pilot", "--agent", "do_nothing", "--seeds", "1", "--no-diagnostics", "--no-grade"], tmp_path / "pilot")
    scenarios = anchors.load_scenarios()["scenarios"]
    assert len(rows) == len(scenarios) == meta["episodes_planned"] and all(r["error"] == "" for r in rows), [r["error"] for r in rows]
    assert meta["pilot"]["assumptions"] == anchors.ASSUMPTIONS and meta["pilot"]["date0"] == "2022-07-01" and meta["grade_lookup"] is False
    cycles = load_cycles(str(default_facts_path()))
    blocks = build_blocks(cycles)
    for scenario, row in zip(scenarios, rows):
        name = f"scenario{scenario['id']}_seed000000"
        assert row["scenario_id"] == str(scenario["id"]) and row["date"] == scenario["date"]
        anchored = anchors.anchor(scenario["date"], cycles, blocks)
        oracle = read_json(run_dir / name / HIDDEN / "oracle.json")
        assert [oracle["diagnosis"][c] for c in COMPONENTS] == anchored["combination"]
        assert json.loads(row["block"]) == anchored["combination"] and int(row["n_faulty"]) == anchored["n_faulty"]
        assert meta["pilot"]["anchors"][name]["cycle_index"] == anchored["cycle_index"]
        report = (run_dir / name / "report.md").read_text(encoding="utf-8")
        assert "## Pilot assumptions" in report and "A1 (mapping)" in report and "A2 (dates)" in report and scenario["date"] in report
        question = (run_dir / name / "question.txt").read_text(encoding="utf-8")
        assert scenario["text"] in question and scenario["date"] in question
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Pilot assumptions" in summary


def test_options_land_in_the_config_and_the_price_template_is_refused(tmp_path):
    run_dir, rows, meta = _run(["--config", "static", "--agent", "do_nothing", "--seeds", "1", "--no-diagnostics", "--clock-mode", "instant",
                                "--guidance", "3", "--verifier-scope", "all", "--prices", "full_tier_high", "--no-grade",
                                "--abstention-rule", "fixed_cut", "--abstention-cut", "0.9"], tmp_path / "options")
    config = meta["config"]
    assert config["clock_mode"] == "instant" and config["guidance_level"] == 3 and config["verifier_scope"] == "all" and config["diagnostics_server"] is False
    assert config["prices"]["read"]["tier3"] == 100 and meta["prices"] == "full_tier_high"
    assert meta["abstention_rule"] == "fixed_cut" and meta["abstention_cut"] == 0.9
    assert rows[0]["clock_mode"] == "instant" and rows[0]["guidance_level"] == "3" and rows[0]["verifier_scope"] == "all" and rows[0]["prices"] == "full_tier_high"
    released = read_json(run_dir / "seed000000" / "released.json")
    assert released["guidance_level"] == 3 and released["rules"]["clock_mode"] == "instant" and released["rules"]["prices"]["read"]["tier3"] == 100
    assert "Procedure:" in (run_dir / "seed000000" / "question.txt").read_text(encoding="utf-8")
    oracle = read_json(run_dir / "seed000000" / HIDDEN / "oracle.json")
    assert "reads" in oracle and read_json(run_dir / "seed000000" / HIDDEN / "score.json")["grade_source"] == "disabled"
    with pytest.raises(SystemExit, match="empty values"):
        run_episodes.main(["--config", "static", "--agent", "do_nothing", "--prices", "sme", "--out", str(tmp_path / "sme")])
