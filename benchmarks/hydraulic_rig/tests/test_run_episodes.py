"""The driver through the two MCP servers: the scripted plan reaches every tool of both servers, the model agent does the
same through the real OpenAI-compatible client against a local stand-in endpoint, and a reference policy played through
the action server scores exactly as it does in process. All need the mcp package of the runtime."""
import csv
import importlib.util
import json
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from common import scenario_env_kwargs
from envs.hydraulic_rig.classifiers import TierClassifiers
from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.env import TinyHydraulicEnv
from envs.hydraulic_rig.greedy import GreedyPolicy
from envs.hydraulic_rig.policies import run_episode
from envs.hydraulic_rig.scoring import score

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package (benchmarks/hydraulic_rig/requirements-smoke.txt)", allow_module_level=True)

import run_episodes  # noqa: E402


def _rows(run_dir):
    with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _only_run(out):
    runs = [p for p in Path(out).iterdir() if p.is_dir()]
    assert len(runs) == 1
    return runs[0]


def test_scripted_plan_reaches_both_servers_and_writes_every_file(tmp_path):
    run_episodes.main(["--scenario", "S1", "--agent", "scripted", "--per-record", "1", "--max-records", "1", "--out", str(tmp_path)])
    run_dir = _only_run(tmp_path)
    rows = _rows(run_dir)
    assert len(rows) == 1 and rows[0]["agent"] == "scripted" and rows[0]["error"] == ""
    assert rows[0]["causal_calls"] == "7" and rows[0]["calls_reaching_no_tool"] == "0" and rows[0]["plan_steps"] == "15"
    ep_dir = run_dir / f"seed{int(rows[0]['seed']):06d}"
    for name in ("episode.json", "released.json", "predictions.json", "trajectory.json", "log.json", "score.json", "plan.txt", "answer.txt", "llm_calls.json", "report.md", "question.txt"):
        assert (ep_dir / name).exists(), name
    trajectory = json.loads((ep_dir / "trajectory.json").read_text(encoding="utf-8"))
    assert len(trajectory) == 15 and all(t["error"] is None for t in trajectory)
    assert {(t["server"], t["tool"]) for t in trajectory} == ({("hydraulic_rig", n) for n in run_episodes.RIG_TOOLS}
                                                             | {("causal_diagnosis", n) for n in run_episodes.CAUSAL_TOOLS})
    predictions = json.loads((ep_dir / "predictions.json").read_text(encoding="utf-8"))
    assert [p["id"] for p in predictions] == ["pred/1"] and predictions[0]["action"] == "replace:pump"
    assert "## Tool calls" in (ep_dir / "report.md").read_text(encoding="utf-8")
    assert (run_dir / "summary.md").exists() and (run_dir / "run.json").exists()
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["servers"] == ["hydraulic_rig", "causal_diagnosis"] and meta["tools_note"] == run_episodes.TOOLS_NOTE_VERSION


def _stand_in_endpoint():
    """A local OpenAI-compatible endpoint that answers every chat completion with the next reply of the scripted plan,
    so the model agent runs through the real client, request and response path without a model."""
    script = run_episodes.ScriptedBackend()
    script.start_episode()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - the http.server hook name
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            reply = {"id": f"stand-in-{len(requests)}", "object": "chat.completion", "created": int(time.time()), "model": body["model"],
                     "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": script.generate("")}}],
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


def test_model_agent_runs_through_the_openai_compatible_client(monkeypatch):
    # the run folder lies inside the repository, as by default, so the runner starts each server as python -m <dotted path>,
    # and the model id carries a dot and a slash, as router ids such as z-ai/glm-5.3 do; either would break that path
    out = run_episodes.ROOT / "benchmarks" / "hydraulic_rig" / "runs" / f"_pytest_{os.getpid()}"
    server, requests = _stand_in_endpoint()
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        run_episodes.main(["--scenario", "S1", "--agent", "plan_execute", "--model", "vendor/stand-in-1.0", "--base-url", base,
                           "--per-record", "1", "--max-records", "1", "--out", str(out.relative_to(run_episodes.ROOT))])
        _assert_stand_in_run(out, requests, base)
    finally:
        server.shutdown()
        shutil.rmtree(out, ignore_errors=True)


def _assert_stand_in_run(out, requests, base):
    run_dir = _only_run(out)
    assert "." not in run_dir.name and "_vendor_stand-in-1_0_" in run_dir.name
    rows = _rows(run_dir)
    assert len(rows) == 1 and rows[0]["error"] == "" and rows[0]["agent"] == "plan_execute_vendor/stand-in-1.0"
    assert rows[0]["causal_calls"] == "7" and rows[0]["calls_reaching_no_tool"] == "0" and rows[0]["plan_steps"] == "15"
    assert rows[0]["llm_calls"] == str(len(requests)) == "17" and rows[0]["input_tokens"] == str(11 * 17)
    assert all(path == "/v1/chat/completions" and body["model"] == "vendor/stand-in-1.0" for path, body in requests)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["endpoint"] == base and meta["model_id"] == "tokenrouter/vendor/stand-in-1.0" and meta["model"] == "vendor/stand-in-1.0"
    assert "## Tool calls" in (run_dir / f"seed{int(rows[0]['seed']):06d}" / "report.md").read_text(encoding="utf-8")


def test_reference_policy_through_the_server_scores_as_in_process(tmp_path):
    run_episodes.main(["--scenario", "S1", "--agent", "greedy_prior_1to2", "--per-record", "1", "--max-records", "2", "--out", str(tmp_path)])
    run_dir = _only_run(tmp_path)
    rows = _rows(run_dir)
    assert len(rows) == 2 and all(r["error"] == "" for r in rows)
    cycles = load_cycles(str(default_facts_path()))
    blocks = build_blocks(cycles)
    classifiers = TierClassifiers(cycles, blocks)
    for row in rows:
        seed = int(row["seed"])
        env = TinyHydraulicEnv(cycles, blocks, **scenario_env_kwargs("S1"))
        expected = json.loads(json.dumps(score(run_episode(env, GreedyPolicy(classifiers, use_max_faulty=True), seed)), default=str))
        served = json.loads((run_dir / f"seed{seed:06d}" / "score.json").read_text(encoding="utf-8"))
        assert served == expected
