"""The driver through the two MCP servers: the scripted plan reaches every tool of both servers, the reference policies
play through the action server with the served log equal to the shadow's, the model agent and the static agent run
through the real OpenAI compatible client against a local stand-in endpoint, anchored runs anchor every scenario, and the
options land in run.json. Every episode is verified from its log; the tests assert relations, never a recorded value.
Run folders go under pytest's tmp_path, outside the repository."""
from CausalDiagnosis.utils.paths import CONFIGS
import csv
import importlib.util
import inspect
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_TOOLS, RIG_TOOLS
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, HIDDEN, HIDDEN_FILES
from CausalDiagnosis.utils.csv_rows import as_bool
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.frameworks.AssetOpsBench.scenarios import scenario_anchors as anchors

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)
if importlib.util.find_spec("CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier") is None or not hasattr(__import__("CausalDiagnosis.environment.Causal_Diagnosis_Environment.env", fromlist=["x"]), "HydraulicRigEnv"):
    pytest.skip("needs HydraulicRigEnv and the verifier", allow_module_level=True)

from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import BRIEFING_VERSION as AOB_BRIEFING_VERSION  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents import scripted_wiring_check  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents import agent_runs  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.tests.pinned_clock import pinned_preset  # noqa: E402


def test_the_recording_backend_keeps_the_wall_time_span_of_every_call_and_of_a_failed_one():
    """Every model call records its start and its end in wall time beside its seconds, also a call that raised,
    so the drivers can place it between the tool calls of the episode."""
    from llm.base import LLMResult

    class Inner:
        def __init__(self):
            self.fail = False

        def generate_with_usage(self, prompt, temperature=0.0):
            if self.fail:
                raise RuntimeError("down")
            return LLMResult(text="ok")

    inner = Inner()
    backend = run_episodes.RecordingBackend(inner)
    before = time.time()
    backend.generate("first")
    inner.fail = True
    with pytest.raises(RuntimeError):
        backend.generate("second")
    after = time.time()
    assert [c["prompt"] for c in backend.calls] == ["first", "second"] and backend.calls[1]["error"] == "RuntimeError: down"
    assert all(before <= c["started_at"] <= c["ended_at"] <= after for c in backend.calls)
    assert backend.calls[0]["ended_at"] <= backend.calls[1]["started_at"]


def test_the_recording_backend_shows_the_context_readers_of_its_inner_backend_and_only_those_it_has():
    """A backend that counts the context keeps its readers behind the wrapper, read and written live, so the runner's context check,
    its compaction trigger and the reset after the summary turn act on a recorded run; a scripted backend keeps none, so the runner still takes every request as fitting."""
    class Counting:
        longest_reply = 0

        def request_fits(self, prompt):
            return len(prompt.split()) + max(1, self.longest_reply) <= 10

        def request_under_trigger(self, prompt):
            return len(prompt.split()) <= 9

    inner = Counting()
    backend = run_episodes.RecordingBackend(inner)
    board = " ".join(["w"] * 9)
    assert backend.request_under_trigger(board) and backend.request_fits(board)
    assert not backend.request_under_trigger(board + " w") and not backend.request_fits(board + " w")
    inner.longest_reply = 3
    assert backend.longest_reply == 3 and not backend.request_fits(board)
    backend.longest_reply = 0
    assert inner.longest_reply == 0 and "longest_reply" not in vars(backend) and backend.request_fits(board)

    scripted = run_episodes.RecordingBackend(object())
    assert all(getattr(scripted, name, None) is None for name in run_episodes.CONTEXT_READERS)


def test_the_recording_backend_records_or_forwards_every_public_method_of_the_served_backend(monkeypatch):
    """Every public method of the backend that the driver serves a model through is recorded by the wrapper or forwarded by
    it, so a method added to the served backend cannot be hidden from the runner again."""
    monkeypatch.setenv("API_KEY", "stand-in-key")
    served = type(run_episodes.model_backends.router_backend("vendor/stand-in-1.0", "http://127.0.0.1:9/v1/"))
    public = {name for name, _ in inspect.getmembers(served) if not name.startswith("_")}
    hidden = sorted(public - set(vars(run_episodes.RecordingBackend)) - set(run_episodes.CONTEXT_READERS))
    assert not hidden, f"RecordingBackend neither records nor forwards {hidden} of {served.__name__}"


def test_a_served_model_whose_backend_cannot_count_the_context_stops_the_run_before_any_episode(tmp_path, monkeypatch):
    """A backend without the readers of the context stops a run with a served model at its start with a named error, and the
    scripted planner is never checked, since its row serves no model."""
    from llm.base import LLMResult

    class Uncounted:
        model_id = "vendor/uncounted"

        def generate_with_usage(self, prompt, temperature=0.0):
            return LLMResult(text="")

        def request_settings(self, temperature=0.0):
            return {}

    with pytest.raises(run_episodes.ContextNotCounted, match="request_fits and no request_under_trigger and no longest_reply"):
        run_episodes.require_context_count(Uncounted())
    monkeypatch.setattr(run_episodes.model_backends, "router_backend", lambda model, base_url=None: Uncounted())
    with pytest.raises(run_episodes.ContextNotCounted, match="Uncounted cannot count the context"):
        run_episodes.main(["--config", "static", "--agent", "causal", "--no-diagnostics", "--no-grade", "--model", "vendor/uncounted",
                           "--base-url", "http://127.0.0.1:9/v1/",
                           "--seeds", "1", "--out", str(tmp_path / "runs")])
    assert not list(tmp_path.rglob("seed000000"))


def test_a_recorded_episode_past_the_compaction_trigger_compacts_and_without_the_forwarders_it_does_not(tmp_path, monkeypatch):
    """The driver path: the causal agent without the server, through RecordingBackend and the chat route, on
    an endpoint that counts every prompt as 950 tokens of a context of 1000, past the trigger of 900 and within the context. The
    board compacts. The same episode with the forwarders of the wrapper taken out takes every request as under the trigger and
    never compacts, so the first assertion rests on the forwarders."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import VERIFYING_READ, ScriptedPlanner, _planner_endpoint, plan_json

    def plans(prompt, call):
        if "No reading yet." in prompt:
            return plan_json([dict(VERIFYING_READ)], mode="test_first")
        return json.dumps({"branch": 1, "mode": "act_now", "why": "scripted", "candidates": [{"levels": {}, "failure_mode_ids": []}],
                           "steps": [{"kind": "diagnose", "args": {"rank": 1}, "expect": {}, "evidence": []}, {"kind": "finish"}]})

    def compactions(out):
        server, _requests = _planner_endpoint(ScriptedPlanner(plans), context=1000, count=950)
        try:
            run_episodes.main(["--config", "static", "--agent", "causal", "--no-diagnostics", "--model", "vendor/stand-in-1.0",
                               "--base-url", f"http://127.0.0.1:{server.server_address[1]}/v1/", "--seeds", "1", "--out", str(out)])
        finally:
            server.shutdown()
        run_dir = next(p for p in out.iterdir() if p.is_dir())
        return read_json(run_dir / "seed000000" / "plans.json")["board"]["compactions"]

    def hidden(self, name):
        raise AttributeError(name)

    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    assert compactions(tmp_path / "forwarded") >= 1
    monkeypatch.setattr(run_episodes.RecordingBackend, "__getattr__", hidden)
    assert compactions(tmp_path / "hidden") == 0


def _rows(run_dir):
    with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run(args, out):
    """One run into a fresh folder under tmp_path; returns the run folder, the rows of scores.csv and run.json."""
    run_episodes.main([*args, "--out", str(out)])
    runs = [p for p in Path(out).iterdir() if p.is_dir()]
    assert len(runs) == 1
    return runs[0], _rows(runs[0]), read_json(runs[0] / "run.json")


PUBLIC_JSON = ("released.json", "predictions.json", "trajectory.json", "llm_calls.json")
PUBLIC_TEXT = ("question.txt", "plan.txt", "answer.txt", "report.md")


def _values(node):
    """Every value of a JSON tree, the containers included, so a drawn structure is found however deep it sits."""
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _values(value)


def _assert_no_hidden_draw(ep_dir):
    """The draws of the hidden samplers, read from the reset entry of the hidden log, reach no public file of the episode:
    the start combination and the scheduled transitions (the per component list and each record with its drawn cycle). A
    public card may share a name with a draw (the schedule card), so the check is on the drawn values and
    never on a word. A bare cycle is also the cycle of the agent's own reads, so a drawn cycle counts as leaked when a
    public object pairs it with the component it was drawn for."""
    reset = read_json(ep_dir / HIDDEN / "log.json")[0]
    assert reset["event"] == "reset" and {"hidden", "schedule"} <= set(reset)
    transitions = list(reset["schedule"])
    draws = [reset["hidden"], *([reset["schedule"]] if reset["schedule"] else []), *transitions]
    for name in PUBLIC_JSON:
        for value in _values(read_json(ep_dir / name)):
            assert value not in draws, (name, value)
            if isinstance(value, dict):
                for transition in transitions:
                    assert not (transition["t"] in value.values() and transition["component"] in value.values()), (name, value)
    text = "".join((ep_dir / name).read_text(encoding="utf-8") for name in PUBLIC_TEXT)
    for draw in draws:
        assert json.dumps(draw) not in text and json.dumps(draw, separators=(",", ":")) not in text, draw


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
    assert len(trajectory) == len(scripted_wiring_check.ScriptedBackend.STEPS) and all(t["error"] is None for t in trajectory)
    assert {(t["server"], t["tool"]) for t in trajectory} == {("hydraulic_rig", n) for n in RIG_TOOLS} | {("causal_diagnosis", n) for n in CAUSAL_TOOLS}
    predictions = json.loads((ep_dir / "predictions.json").read_text(encoding="utf-8"))
    assert [p["id"] for p in predictions if "branch" not in p] == ["pred/1"]  # the troubleshooting plans of the ranking store their own records
    # the declaration after the diagnosis is the one refused call of the plan
    assert rows[0]["refusals"] == "1" and rows[0]["tool_calls"] == str(sum(1 for t in trajectory if t["server"] == "hydraulic_rig"))
    verdict = read_json(ep_dir / HIDDEN / "verdict.json")
    assert verdict["verified"] and rows[0]["pass"] == str(verdict["pass"]) and rows[0]["end_reason"] == "finish"
    assert rows[0]["cost_oracle"] != "" and rows[0]["reward"] != ""
    # static starts lie in the label table, so the independent label was looked up with the server's E18 grade inside it
    score = read_json(ep_dir / HIDDEN / "score.json")
    assert score["grade_source"] == "table" and meta["grade_lookup"] is True
    assert set(verdict["grade"]) == {"separable", "share", "rollouts", "source", "rules_key", "e18"} and verdict["separable"] is verdict["grade"]["separable"]
    assert verdict["grade"]["e18"] is None or set(verdict["grade"]["e18"]) == {"separable", "commit_share", "first_share", "cost"}
    # the public files hold no draw of the hidden samplers and the report no hidden value
    _assert_no_hidden_draw(ep_dir)
    report = (ep_dir / "report.md").read_text(encoding="utf-8")
    oracle = read_json(ep_dir / HIDDEN / "oracle.json")
    assert "## Tool calls" in report and json.dumps([oracle["diagnosis"][c] for c in COMPONENTS]) not in report
    assert "verdict" not in report.replace("the verdict and the scores", "")
    assert meta["servers"] == ["hydraulic_rig", "causal_diagnosis"] and meta["briefing"] == run_episodes.BRIEFING_VERSION and meta["mode"] == "gym"
    assert meta["config"]["clock_mode"] == "declared" and meta["episodes_planned"] == 1 and meta["anchored"] is None
    assert meta["abstention_rule"] == "decision" and meta["abstention_cut"] is None
    # the recorded ranking model is the one the diagnostics server reports in its ranking answer, and both are the server's default
    ranked = next(t for t in trajectory if t["tool"] == "rank_root_causes")
    reported = json.loads(ranked["response"])["ranking_model"]
    assert meta["ranking_model"] == rows[0]["ranking_model"] == reported == run_episodes.DEFAULT_RANKING_MODEL
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Episodes planned 1, episode folders 1, rows in scores.csv 1" in summary and "| hidden/verdict.json | 1 |" in summary


def test_the_public_files_of_a_timed_episode_hold_no_drawn_transition(tmp_path):
    """The timed preset draws the transitions of one component, and the guard on the hidden draws holds on such an episode as
    on a static one."""
    run_dir, rows, _meta = _run(["--config", "timed", "--agent", "scripted", "--seeds", "1"], tmp_path / "timed")
    assert len(rows) == 1 and rows[0]["error"] == "", rows[0]["error"]
    ep_dir = run_dir / "seed000000"
    reset = read_json(ep_dir / HIDDEN / "log.json")[0]
    assert reset["schedule"] and all({"t", "component"} <= set(drop) for drop in reset["schedule"])  # the draw the guard must catch
    _assert_no_hidden_draw(ep_dir)


def test_scripted_plan_runs_the_base_row_through_the_assetopsbench_servers(tmp_path):
    """--servers aob: the launchers of iot, fmsr, wo and the task server serve AssetOpsBench's own tools and the task
    tools behind the metering layer over the memory store, the driver creates the episode's databases before the first
    call and exports them after the verifier, the question is the briefing of the rows, and run.json records the store."""
    run_dir, rows, meta = _run(["--config", "base", "--agent", "scripted", "--seeds", "1", "--servers", "aob"], tmp_path / "aob")
    assert len(rows) == 1 and rows[0]["error"] == "", rows[0]["error"]
    assert meta["servers"] == ["iot", "fmsr", "wo", "task", "causal_diagnosis"] and meta["backend"] == {"kind": "memory"}
    assert meta["briefing"] == AOB_BRIEFING_VERSION != run_episodes.BRIEFING_VERSION  # the briefing of the rows, not the rig's
    ep_dir = run_dir / "seed000000"
    for name in ("iot_server.py", "fmsr_server.py", "wo_server.py", "task_server.py", "causal_diagnosis_server.py", "aob_documents.json"):
        assert (ep_dir / name).exists(), name
    assert not (ep_dir / "aob_store.json").exists()  # the memory store is dropped with the databases
    question = (ep_dir / "question.txt").read_text(encoding="utf-8")
    assert "hyd_1" in question and "read_sensors" not in question  # the briefing of the rows, not the rig's
    trajectory = json.loads((ep_dir / "trajectory.json").read_text(encoding="utf-8"))
    steps = scripted_wiring_check.ScriptedBackend.AOB_STEPS
    assert [(t["server"], t["tool"]) for t in trajectory] == [(server, tool) for _, server, tool, _ in steps] and all(t["error"] is None for t in trajectory)
    state = read_json(ep_dir / "episode.json")
    assert [c["server"] for c in state["tool_calls"]] == [server for _, server, _, _ in steps if server != "causal_diagnosis"]
    assert [a["type"] for a in state["actions"]] == ["get_asset_profile", "aob_read", "get_failure_modes", "read_recorded", "read_recorded",
                                                     "submit_diagnosis", "create_work_order", "aob_read", "aob_read", "finish"]
    assert all(c["price"] == 0 for c in state["tool_calls"]) and not any(c.get("refused") for c in state["tool_calls"])  # the base row charges nothing
    assert rows[0]["refusals"] == "0" and rows[0]["tool_calls"] == "10" and rows[0]["causal_calls"] == "2" and rows[0]["end_reason"] == "finish"
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import episode_start_and_replay as aob_rows
    prehistory = [t for t in aob_rows.replay(state).recorded_cycles() if t < 0]  # the start window, recorded before the first call
    released = read_json(ep_dir / "released.json")
    assert prehistory and [o["t"] for o in released["observations"]][:5] == prehistory[:5]  # the history page, served from the record
    documents = read_json(ep_dir / "aob_documents.json")
    assert {d["assetnum"] for d in documents["asset"]} == {"hyd_1", "hyd_1_cooler", "hyd_1_valve", "hyd_1_pump", "hyd_1_accumulator"}
    assert [(d["wonum"], d["status"]) for d in documents["workorder"]] == [("wo-1", "COMP")]  # closed inside its call without a clock
    assert len(documents["iot"]) == len(prehistory) and all("_rev" not in d for d in documents["iot"])  # no cycle plays without a clock
    verdict = read_json(ep_dir / HIDDEN / "verdict.json")
    assert verdict["verified"] and rows[0]["pass"] == str(verdict["pass"])


def test_starts_combinations_plays_every_allowed_combination_under_its_smallest_seed(tmp_path):
    allowed = [[100, 100, 0, 90], [3, 100, 0, 130], [3, 73, 0, 130]]
    config = {**read_json(CONFIGS / "development" / "static.json"), "name": "three", "allowed": allowed}
    (tmp_path / "three.json").write_text(json.dumps(config), encoding="utf-8")
    run_dir, rows, meta = _run(["--config", str(tmp_path / "three.json"), "--agent", "do_nothing", "--seeds", "1", "--starts", "combinations"], tmp_path / "combos")
    assert meta["starts"] == "combinations" and meta["episodes_planned"] == 3 and len(rows) == 3
    assert sorted(set(meta["seeds"])) == sorted(meta["seeds"]) and [int(r["seed"]) for r in rows] == meta["seeds"]
    truths = [read_json(run_dir / f"seed{int(r['seed']):06d}" / HIDDEN / "oracle.json")["diagnosis"] for r in rows]
    assert sorted(tuple(t[c] for c in COMPONENTS) for t in truths) == sorted(map(tuple, allowed))


def test_starts_sampled_plays_one_replicate_per_run_and_run_json_records_it_with_its_rule(tmp_path):
    """The run unit of the grid: a run of --starts sampled plays one replicate of the test set,
    every allowed start once at the start state of that replicate, replicate one by default, so a line of the grid is five runs
    that each fit a job, and run.json records the replicate and the rule that chose its seeds. A run of other starts records
    neither, and --replicate needs --starts sampled."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    allowed = [[100, 100, 0, 90], [3, 100, 0, 130], [3, 73, 0, 130]]
    config = {**read_json(CONFIGS / "development" / "static.json"), "name": "three", "allowed": allowed}
    (tmp_path / "three.json").write_text(json.dumps(config), encoding="utf-8")
    line = ["--config", str(tmp_path / "three.json"), "--agent", "do_nothing", "--starts", "sampled"]
    _first_dir, first, meta_first = _run(line, tmp_path / "first")
    _third_dir, third, meta_third = _run([*line, "--replicate", "3"], tmp_path / "third")
    cycles = load_cycles(default_facts_path())
    env = HydraulicRigEnv(cycles, build_blocks(cycles), load_config(str(tmp_path / "three.json"))[1])
    assert meta_first["seeds"] == start_states.replicate_seeds(env, 1) and meta_third["seeds"] == start_states.replicate_seeds(env, 3)
    assert (meta_first["replicate"], meta_third["replicate"], meta_third["master_seed"], meta_third["seed_pool"]) == (1, 3, 42, 8)
    assert meta_first["replicate_rule"] == meta_third["replicate_rule"] == start_states.REPLICATE_RULE
    assert meta_third["episodes_planned"] == len(third) == 3 and [int(r["seed"]) for r in third] == meta_third["seeds"]
    entry = {r["block"]: r["entry_offset"] for r in first}
    assert sorted(entry) == sorted(r["block"] for r in third) and len(entry) == 3  # every start once per run
    assert all(entry[r["block"]] != r["entry_offset"] for r in third)  # at another entry offset of its block
    _seeds_dir, _rows_seeds, meta_seeds = _run(["--config", "static", "--agent", "do_nothing", "--seeds", "1", "--no-grade"], tmp_path / "seeds")
    assert meta_seeds["replicate"] is None and meta_seeds["replicate_rule"] is None
    parser = run_episodes.build_parser()
    assert parser.parse_args([]).replicate is None and parser.parse_args(["--replicate", "5"]).replicate == 5
    with pytest.raises(SystemExit):
        parser.parse_args(["--replicate", "6"])
    with pytest.raises(SystemExit, match="--starts sampled"):
        run_episodes.main(["--config", "static", "--agent", "do_nothing", "--replicate", "2", "--out", str(tmp_path / "refused")])


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
        # the oracle diagnoses and passes on every start, whatever the grade says, which is the success the reward follows (RFC 008)
        assert as_bool(oracle["pass"]) and as_bool(oracle["diagnosis_all_correct"]) and not as_bool(oracle["abstained"])
        assert float(oracle["reward"]) > float(nothing["reward"])


def test_the_declare_policy_plays_through_the_action_server_on_the_row_of_the_main_comparison(tmp_path):
    run_dir, rows, meta = _run(["--config", pinned_preset(tmp_path, "real_time"), "--agent", "declare", "--seeds", "2", "--no-diagnostics", "--no-grade"],
                               tmp_path / "declare")
    assert meta["servers"] == ["hydraulic_rig"] and meta["agent"] == meta["agent_label"] == "declare"
    assert all(r["error"] == "" and as_bool(r["replay_matches"]) and r["agent"] == "declare" and r["end_reason"] == "finish" for r in rows), rows
    # without a label every start counts as separable, so the declaration cites its reading and earns nothing
    assert all(as_bool(r["abstained"]) and as_bool(r["abstention_cites_reading"]) and not as_bool(r["abstention_correct"]) for r in rows)
    assert all(not as_bool(r["pass"]) and not as_bool(r["success"]) and r["failure_class"] == "abstention_on_separable" for r in rows)
    log = read_json(run_dir / "seed000000" / HIDDEN / "log.json")
    assert [e["event"] for e in log if e.get("step") is not None] == ["read", "declare_insufficient_evidence", "finish"]


STATIC_REPLY = "The pump is faulty and should be replaced; the cooler is healthy."


def _inseparable_seed(tmp_path, limit=60):
    """The first seed of the static preset whose start the label table marks inseparable, or None within the limit."""
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    facts_path, cycles, blocks = load_world()
    _, config = load_config("static")
    tmp_path.mkdir(parents=True, exist_ok=True)
    episode.start(tmp_path / "episode.json", 0, config, facts_path)
    released = read_json(tmp_path / "released.json")
    env = HydraulicRigEnv(cycles, blocks, config)
    for seed in range(limit):
        env.reset(seed)
        grade, source = run_episodes.grade_of_start(env, released)  # the table the runs use
        if source == "table" and grade is not None and not grade["separable"]:
            return seed
    return None


def test_the_oracle_diagnoses_on_an_inseparable_start_and_outranks_do_nothing(tmp_path):
    seed = _inseparable_seed(tmp_path / "probe")
    if seed is None:
        pytest.skip("no inseparable start among the first seeds of the static preset under the default grade table")
    rows = {}
    for agent in ("oracle", "do_nothing"):
        _run_dir, agent_rows, _meta = _run(["--config", "static", "--agent", agent, "--seeds", "1", "--first-seed", str(seed), "--no-diagnostics"], tmp_path / agent)
        assert agent_rows[0]["error"] == "" and as_bool(agent_rows[0]["replay_matches"])
        rows[agent] = agent_rows[0]
    oracle, nothing = rows["oracle"], rows["do_nothing"]
    assert oracle["separable"] == "False" and nothing["separable"] == "False"
    # the oracle diagnoses, repairs every faulty component, and is a success on this start as on every other
    assert as_bool(oracle["pass"]) and as_bool(oracle["success"]) and not as_bool(oracle["abstained"])
    assert as_bool(oracle["recovered"]) and as_bool(oracle["commitment_on_inseparable"])
    assert not as_bool(nothing["success"]) and float(oracle["reward"]) > float(nothing["reward"])


def test_a_label_lookup_that_fails_leaves_the_error_in_the_row_instead_of_scoring_the_episode_as_separable(tmp_path, monkeypatch):
    """A label or grade lookup that raises is a verifier that could not run: the row carries the error with its scores empty,
    and the episode is never scored under an unknown label, which would read every declaration as an abstention on a separable
    start with no visible reason."""
    from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability

    def broken(env, path=None):
        raise PermissionError("labels.json is not writable")

    monkeypatch.setattr(separability, "label_of", broken)
    _run_dir, rows, _meta = _run(["--config", "static", "--agent", "declare", "--seeds", "1", "--no-diagnostics"], tmp_path / "broken")
    assert "PermissionError: labels.json is not writable" in rows[0]["error"]
    assert rows[0]["pass"] == "" and rows[0]["success"] == "" and rows[0]["separable"] == "" and rows[0]["failure_class"] == ""


def test_the_e18_record_takes_the_majority_of_first_commitments_and_a_start_no_rollout_commits_on_is_not_separable(monkeypatch):
    """The binary label of E18 reads the rule of the independent label, so a start on which every rollout ends on the truth and
    most begin elsewhere is not separable, a tie is no majority, and a start without a grade is not separable with both shares 0. A
    start the grade table does not hold raises, as a failed label lookup does, and models fitted on another data
    file than the one the environment plays are refused."""
    from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import tools as adapter
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as grade

    label = {"separable": True, "share": 1.0, "rollouts": 10, "source": "greedy reference", "rules_key": "ab" * 32}
    grades = {"1,1": None, "2,2": {"commit_share": 1.0, "first_share": 0.4, "cost": 9.0}, "3,3": {"commit_share": 0.7, "first_share": 0.5, "cost": 8.0},
              "4,4": {"commit_share": 0.7, "first_share": 0.6, "cost": 7.0}}
    monkeypatch.setattr(separability, "label_of", lambda env: label)
    monkeypatch.setattr(adapter.fitted_models, "load_models", lambda: SimpleNamespace(raw={"facts_sha256": "c" * 64}))
    monkeypatch.setattr(grade, "ensure_grades", lambda *args: {"grades": grades})

    def e18_of(start, facts_sha256="c" * 64):
        record, source = run_episodes.grade_of_start(SimpleNamespace(config=SimpleNamespace(asset_id="hyd_1"), initial=start, facts_sha256=facts_sha256), {"rules": {}})
        assert source == "table" and record["separable"] is True
        return record["e18"]

    assert e18_of((1, 1)) == {"separable": False, "commit_share": 0.0, "first_share": 0.0, "cost": None}
    assert e18_of((2, 2)) == {"separable": False, "commit_share": 1.0, "first_share": 0.4, "cost": 9.0}
    assert e18_of((3, 3))["separable"] is False and e18_of((4, 4))["separable"] is True
    with pytest.raises(LookupError):
        e18_of((5, 5))
    with pytest.raises(records.DataFileError):
        e18_of((2, 2), "d" * 64)


def first_seed_of_each_e18_verdict(config_name, limit=64):
    """{E18 verdict: the smallest seed of the preset whose start the E18 of the tree marks so}, read from the grade table of the
    row's rules by the driver's own lookup, so that a test of both verdicts names no seed and no combination. A preset whose
    first limit seeds hold no start of one verdict fails the test by name."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.environment.reference_policies.diagnosability_label import start_grade

    cycles = load_cycles(default_facts_path())
    blocks, config, first = build_blocks(cycles), load_config(config_name)[1], {}
    for seed in range(limit):
        env = HydraulicRigEnv(cycles, blocks, config)
        env.reset(seed)
        grade = start_grade(env)
        if grade is not None and grade.get("e18") is not None:
            first.setdefault(bool(grade["e18"]["separable"]), seed)
        if len(first) == 2:
            return first
    pytest.fail(f"the first {limit} seeds of {config_name} hold no start of each E18 verdict, only {sorted(first)}")


def test_a_declaration_on_a_start_the_label_marks_inseparable_earns_nothing_where_e18_marks_it_separable(tmp_path, monkeypatch):
    """The credit rule of a declaration through the driver: with the label of every start set to inseparable, a declaration
    that cites its reading succeeds exactly on the starts that E18, read from the grade table of the row's rules, marks
    inseparable too, and elsewhere its failure class names E18. The test plays the first seed of the static preset whose start
    E18 marks separable and the first whose start it marks inseparable."""
    from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability

    first = first_seed_of_each_e18_verdict("static")
    label_of = separability.label_of
    monkeypatch.setattr(separability, "label_of", lambda env: {**label_of(env), "separable": False})
    kinds = set()
    for seed in sorted(first.values()):
        run_dir, rows, _meta = _run(["--config", "static", "--agent", "declare", "--first-seed", str(seed), "--seeds", "1", "--no-diagnostics"],
                                    tmp_path / f"declare{seed}")
        for row in rows:
            grade = read_json(run_dir / f"seed{int(row['seed']):06d}" / HIDDEN / "verdict.json")["grade"]
            solved = grade["e18"]["separable"]
            kinds.add(solved)
            assert row["error"] == "" and grade["separable"] is False and as_bool(row["abstention_cites_reading"]), row
            assert as_bool(row["abstention_correct"]) is as_bool(row["success"]) is (not solved)
            assert row["failure_class"] == ("abstention_on_e18_separable" if solved else "none")
    assert kinds == {True, False}


def test_a_drawn_start_missing_from_the_label_or_the_grade_table_leaves_the_error_in_the_row_instead_of_a_score(tmp_path, monkeypatch):
    """A grade is unknown, with no credit, only by design, for a start outside the fault rule of the sampler such as an anchored
    day. A drawn start always lies in the fault rule, so a label table or an E18 grade table without it is a failed lookup, and
    the row carries the error with its scores empty, as for any lookup that raises."""
    from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import tools as adapter
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as grade

    label_of = separability.label_of
    args = ["--config", "static", "--agent", "declare", "--seeds", "1", "--no-diagnostics"]
    monkeypatch.setattr(separability, "label_of", lambda env: None)
    _run_dir, rows, _meta = _run(args, tmp_path / "no_label")
    assert "LookupError" in rows[0]["error"] and "label table" in rows[0]["error"], rows[0]["error"]
    assert rows[0]["success"] == "" and rows[0]["separable"] == "" and rows[0]["reward"] == ""
    outside = SimpleNamespace(config=SimpleNamespace(asset_id="hyd_1"), initial=(3, 73, 2, 90), candidates=[(3, 100, 0, 130), (100, 90, 0, 130)])
    assert run_episodes.grade_of_start(outside, {"rules": {}}) == (None, "not in table")
    monkeypatch.setattr(separability, "label_of", label_of)
    monkeypatch.setattr(adapter.fitted_models, "load_models", lambda: SimpleNamespace(raw={"facts_sha256": records.file_sha256(default_facts_path())}))
    monkeypatch.setattr(grade, "ensure_grades", lambda *args, **kwargs: {"grades": {}})
    _run_dir, rows, _meta = _run(args, tmp_path / "no_grade")
    assert "LookupError" in rows[0]["error"] and "E18" in rows[0]["error"], rows[0]["error"]
    assert rows[0]["success"] == "" and rows[0]["separable"] == "" and rows[0]["reward"] == ""


def test_the_driver_refuses_models_fitted_on_another_data_file_than_the_run_reads(tmp_path):
    """A run that reads another data file than the models were fitted on stops before its run folder exists, with the refusal
    that names the command to fit them again, so that no episode is scored with models of other recordings."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import REFIT

    lines = Path(default_facts_path()).read_text(encoding="utf-8").splitlines(keepends=True)
    first = json.loads(lines[0])
    first["features"][0]["value"] += 1.0
    changed, out = tmp_path / "changed.jsonl", tmp_path / "runs"
    changed.write_text(json.dumps(first) + "\n" + "".join(lines[1:]), encoding="utf-8")
    with pytest.raises(records.DataFileError) as refused:
        run_episodes.main(["--config", "static", "--agent", "declare", "--seeds", "1", "--no-diagnostics", "--facts", str(changed),
                           "--out", str(out)])
    assert REFIT in str(refused.value) and records.file_sha256(changed) in str(refused.value) and not out.exists()


def _stand_in_endpoint(fixed_reply=None):
    """A local OpenAI compatible endpoint that answers every chat completion with the next reply of the scripted plan
    (or with one fixed reply), so the model agents run through the real client, request and response path without a
    model, and lists one model without a context length, as a router does, for the judge that reads its served context."""
    script = scripted_wiring_check.ScriptedBackend()
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

        def do_GET(self):  # noqa: N802 - the http.server hook name
            listed = self.path.rstrip("/").endswith("/models")
            data = json.dumps({"object": "list", "data": [{"id": "stand-in", "object": "model"}]} if listed else {}).encode()
            self.send_response(200 if listed else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


class _PlannerOrProse:
    """The causal agent's scripted planner for its own prompts and a line of prose for any other, the judge's."""

    def __init__(self):
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import ScriptedPlanner, act_now

        self.planner = ScriptedPlanner(act_now)

    def generate(self, prompt):
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import ORIENTATION_HEADING, PLANNING_HEADING, SUMMARY_HEADING

        if any(heading in prompt for heading in (ORIENTATION_HEADING, PLANNING_HEADING, SUMMARY_HEADING)):
            return self.planner.generate(prompt)
        return STATIC_REPLY


def test_model_agent_runs_through_the_openai_compatible_client_with_a_judge(tmp_path, monkeypatch):
    """The chat client of the driver with a judge on the same endpoint, played by the causal agent, the one agent on the chat
    route since plan_execute plays the paper-era port."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import _planner_endpoint

    server, requests = _planner_endpoint(_PlannerOrProse())
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        with pytest.raises(SystemExit, match="self judging"):
            run_episodes.main(["--config", "static", "--agent", "causal", "--model", "vendor/stand-in-1.0", "--base-url", base, "--judge", "local",
                               "--out", str(tmp_path / "refused")])
        assert run_episodes.build_parser().get_default("judge_protocol") == "paper"  # the paper's five calls unless asked
        run_dir, rows, meta = _run(["--config", "static", "--agent", "causal", "--model", "vendor/stand-in-1.0", "--base-url", base, "--seeds", "1",
                                    "--judge", "local", "--judge-model", "vendor/stand-in-judge", "--judge-protocol", "single"], tmp_path / "model")
        assert "." not in run_dir.name and "_vendor_stand-in-1_0_" in run_dir.name
        assert len(rows) == 1 and rows[0]["error"] == "" and rows[0]["agent"] == "causal_vendor/stand-in-1.0" and rows[0]["model"] == "vendor/stand-in-1.0"
        agent_calls = [body for _, body in requests if body["model"] == "vendor/stand-in-1.0"]
        # every call of the agent goes to the endpoint, and the judge's one call goes to the same stand in
        assert int(rows[0]["llm_calls"]) == len(agent_calls) >= 3 and len(requests) == len(agent_calls) + 1
        assert rows[0]["input_tokens"] == str(11 * len(agent_calls))
        assert all(path == "/v1/chat/completions" for path, _ in requests)
        assert [body["model"] for _, body in requests].count("vendor/stand-in-judge") == 1
        # run.json names the models and no endpoint: the URL of a machine stays out of every public file
        assert meta["endpoint"] is None and meta["model_id"] == "tokenrouter/vendor/stand-in-1.0" and meta["model"] == "vendor/stand-in-1.0"
        assert meta["judge_model"] == "vendor/stand-in-judge" and meta["judge_endpoint"] is None
        # the stand in answers the judge with prose, not a rubric, so the judge record exists and its one failed call reads false
        judged = read_json(run_dir / "seed000000" / HIDDEN / "judge.json")
        assert judged["protocol"] == "single" and judged["pass"] is False and judged["pass_rate"] == 0.0 and rows[0]["judge_pass"] == "False"
    finally:
        server.shutdown()


def test_the_static_request_names_the_probe_of_the_model_endpoint_so_a_dead_endpoint_ends_the_run(tmp_path, monkeypatch):
    """The static agent plays no episode, so its one model request names the health probe of the model's endpoint
    (judge.endpoint_probe): on an endpoint that no longer answers, the run ends with ServiceDown before the row is written, where
    the failed request once stood in the row as an error of the agent. The control: a first attempt that the service answers
    with 503 while the probe answers is sent again, and the row holds the answer with no error."""
    import types

    from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
    import httpx
    from CausalDiagnosis.utils import model_request_retry as retry

    class Unavailable(Exception):
        status_code = 503  # as the SDK raises an answer of 503

    class PolicyBackend:
        """The model behind the one retry policy of the tree (CausalDiagnosis.utils.model_request_retry.post_with_retries), as the chat backend sends: each
        attempt takes the next of failures, an exception that fails it, and answers the static reply once they are used up."""
        longest_reply, model_id = 0, "tokenrouter/vendor/stand-in-1.0"

        def __init__(self, failures_of_attempts):
            self.failures, self.sent = list(failures_of_attempts), 0

        def request_fits(self, prompt):
            return True

        request_under_trigger = request_fits

        def generate_with_usage(self, prompt, temperature=0.0):
            def send(timeout):
                self.sent += 1
                if self.failures:
                    raise self.failures.pop(0)
                return types.SimpleNamespace(text=STATIC_REPLY, input_tokens=11, output_tokens=7, finish_reason="stop")

            return retry.post_with_retries(send)

    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)
    health = {"status": None}
    monkeypatch.setattr(failures, "endpoint_health", lambda base_url, key, timeout_s: {"ok": health["status"] is not None, **health})
    args = ["--config", "static", "--agent", "static", "--model", "vendor/stand-in-1.0", "--base-url", "http://127.0.0.1:1/v1/",
            "--seeds", "1", "--no-grade"]
    dead = PolicyBackend([httpx.ConnectError("refused")] * 4)
    monkeypatch.setattr(run_episodes.model_backends, "router_backend", lambda model, base_url: dead)
    with pytest.raises(retry.ServiceDown, match="no longer answers its health probe"):
        run_episodes.main([*args, "--out", str(tmp_path / "dead")])
    assert dead.sent == 1 and not list((tmp_path / "dead").glob("*/scores.csv"))
    health["status"] = 200
    live = PolicyBackend([Unavailable()])
    monkeypatch.setattr(run_episodes.model_backends, "router_backend", lambda model, base_url: live)
    _run_dir, rows, _meta = _run(args, tmp_path / "live")
    assert len(rows) == 1 and rows[0]["error"] == "" and rows[0]["llm_calls"] == "1" and live.sent == 2, rows


def test_static_agent_answers_without_the_gym_and_is_judged_and_parsed(tmp_path, monkeypatch):
    server, requests = _stand_in_endpoint(fixed_reply=STATIC_REPLY)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        # every reply of the stand in is one prose sentence, the static answer of both episodes and the judge's five replies under
        # the default protocol of the paper, which are no rubric, so every call fails and the judge's pass reads False
        run_dir, rows, meta = _run(["--config", "static", "--agent", "static", "--model", "vendor/stand-in-1.0", "--base-url", base, "--seeds", "2",
                                    "--no-grade", "--judge", "local", "--judge-model", "vendor/stand-in-judge"], tmp_path / "static")
        assert meta["mode"] == "static" and meta["agent"] == "static" and len(rows) == 2 and all(r["error"] == "" for r in rows), [r["error"] for r in rows]
        for row in rows:
            ep_dir = run_dir / f"seed{int(row['seed']):06d}"
            assert row["end_reason"] == "static" and row["tool_calls"] == "0" and row["pass"] == "" and row["llm_calls"] == "1"
            assert row["static_faulty_set_correct"] in ("True", "False") and row["judge_pass"] == "False"
            for name in ("prompt.txt", "answer.txt", "question.txt", "llm_calls.json", "report.md", "released.json", "episode.json"):
                assert (ep_dir / name).exists(), name
            for name in ("oracle.json", "log.json", "static.json", "judge.json", "score.json"):
                assert (ep_dir / HIDDEN / name).exists(), name
            static = read_json(ep_dir / HIDDEN / "static.json")
            oracle = read_json(ep_dir / HIDDEN / "oracle.json")
            assert static["features_shown"] > 0 and static["failure_modes_shown"] > 0
            assert static["truth_faulty"] == [c for c in COMPONENTS if oracle["diagnosis"][c] != HEALTHY[c]]
            prompt = (ep_dir / "prompt.txt").read_text(encoding="utf-8")
            assert "You cannot call any tool" in prompt and "Failure modes of the asset" in prompt and "Pressure_PS1_mean" in prompt
            # the question states the step cap, the step of a turn without a call and the wall limit, the static prompt none of them
            question = (ep_dir / "question.txt").read_text(encoding="utf-8")
            rules = ("Step cap: ", "still takes one step", "If the wall limit ends the episode")
            assert all(rule in question for rule in rules) and not any(rule in prompt for rule in rules)
            # the facts shown are features only: no label of the facts file and no hidden level reaches the prompt
            assert "labels" not in prompt and "failure_profile" not in prompt and json.dumps(oracle["diagnosis"]) not in prompt
            report = (ep_dir / "report.md").read_text(encoding="utf-8")
            assert "## Static mode" in report and "which the gym charges for" in report and "## Static answer" in report
            assert "trajectory.json" not in [p.name for p in ep_dir.iterdir()]
            parsed = read_json(ep_dir / HIDDEN / "static.json")["parsed"]
            assert parsed["faulty"] == ["pump"] and parsed["repaired"] == ["pump"] and parsed["sentences"]["cooler"] == []
            judge = read_json(ep_dir / HIDDEN / "judge.json")
            assert judge["mode"] == "static" and judge["answer"] == STATIC_REPLY and judge["pass"] is False
            assert judge["protocol"] == "paper" and judge["calls_planned"] == len(judge["calls"]) == 5 and judge["pass_rate"] == 0.0
        # one model call per episode and the five judge calls of the paper protocol per episode reached the stand in
        assert len(requests) == 12
    finally:
        server.shutdown()


@pytest.mark.skipif(not anchors.scenarios_path().is_file(), reason=f"the scenarios are not fetched; run {anchors.FETCH_COMMAND}")
def test_the_anchored_preset_anchors_every_scenario_at_its_date_and_prints_the_assumptions(tmp_path):
    run_dir, rows, meta = _run(["--config", "anchored", "--agent", "do_nothing", "--seeds", "1", "--no-diagnostics", "--no-grade"], tmp_path / "anchored")
    scenarios = anchors.load_scenarios()["scenarios"]
    assert len(rows) == len(scenarios) == meta["episodes_planned"] and all(r["error"] == "" for r in rows), [r["error"] for r in rows]
    assert meta["anchored"]["assumptions"] == anchors.ASSUMPTIONS and meta["anchored"]["date0"] == "2022-07-01" and meta["grade_lookup"] is False
    cycles = load_cycles(str(default_facts_path()))
    blocks = build_blocks(cycles)
    for scenario, row in zip(scenarios, rows):
        name = f"scenario{scenario['id']}_seed000000"
        assert row["scenario_id"] == str(scenario["id"]) and row["date"] == scenario["date"]
        anchored = anchors.anchor(scenario["date"], cycles, blocks)
        oracle = read_json(run_dir / name / HIDDEN / "oracle.json")
        assert [oracle["diagnosis"][c] for c in COMPONENTS] == anchored["combination"]
        assert json.loads(row["block"]) == anchored["combination"] and int(row["n_faulty"]) == anchored["n_faulty"]
        assert meta["anchored"]["anchors"][name]["cycle_index"] == anchored["cycle_index"]
        report = (run_dir / name / "report.md").read_text(encoding="utf-8")
        assert "## Anchored scenario assumptions" in report and "A1 (mapping)" in report and "A2 (dates)" in report and scenario["date"] in report
        question = (run_dir / name / "question.txt").read_text(encoding="utf-8")
        assert scenario["text"] in question and scenario["date"] in question
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Anchored scenario assumptions" in summary


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


def test_the_diagnostics_launcher_alone_names_the_start_s_block_and_no_file_the_agent_reads_does(tmp_path):
    """The launcher of the diagnostics server names the combination the episode started in, whose block every fit
    of the server leaves out, the action server's launcher does not, and no file the agent reads names it."""
    run_dir, rows, meta = _run(["--config", "static", "--agent", "scripted", "--seeds", "1", "--no-grade"], tmp_path / "rig")
    assert rows[0]["error"] == "" and "diagnostics_table" not in COLUMNS and "diagnostics_table" not in meta
    episode = run_dir / "seed000000"
    start = tuple(read_json(episode / HIDDEN / "log.json")[0]["hidden"])
    assert f"left_out={start!r}" in (episode / "causal_diagnosis_server.py").read_text(encoding="utf-8")
    assert "left_out" not in (episode / "hydraulic_rig_server.py").read_text(encoding="utf-8")
    _assert_no_hidden_draw(episode)
    public = [episode / name for name in (*PUBLIC_JSON, *PUBLIC_TEXT) if (episode / name).exists()]
    assert public and all("left_out" not in path.read_text(encoding="utf-8") for path in public)


def test_the_diagnostics_launcher_names_the_start_s_block_over_the_assetopsbench_servers(tmp_path):
    """The engine of the AssetOpsBench rows reaches the launcher with the same start, and no other launcher and no trajectory names it."""
    run_dir, rows, meta = _run(["--config", pinned_preset(tmp_path, "real_time"), "--agent", "scripted", "--seeds", "1", "--no-grade",
                                "--servers", "aob"], tmp_path / "aob")
    assert rows[0]["error"] == ""
    episode = run_dir / "seed000000"
    start = tuple(read_json(episode / HIDDEN / "log.json")[0]["hidden"])
    assert f"left_out={start!r}" in (episode / "causal_diagnosis_server.py").read_text(encoding="utf-8")
    assert all("left_out" not in (episode / f"{server}_server.py").read_text(encoding="utf-8") for server in ("iot", "fmsr", "wo", "task"))
    trajectory = read_json(episode / "trajectory.json")
    assert any(t["tool"] == "rank_root_causes" for t in trajectory) and all(t["error"] is None for t in trajectory)
    assert "left_out" not in json.dumps(trajectory) and "left_out" not in (episode / "released.json").read_text(encoding="utf-8")


def test_the_driver_takes_no_table_of_the_diagnostics_server():
    with pytest.raises(SystemExit):
        run_episodes.build_parser().parse_args(["--diagnostics-table", "served"])


def test_two_runs_of_one_label_get_distinct_folders(tmp_path):
    first = run_episodes.run_folder(str(tmp_path), "static_oracle_20261102-000000")
    second = run_episodes.run_folder(str(tmp_path), "static_oracle_20261102-000000")
    assert first.is_dir() and second.is_dir() and first != second and second.name == first.name + "-2"


def test_every_record_function_counts_every_attempted_diagnostics_call_whatever_its_error(tmp_path):
    """The steps of the execution efficiency add causal_calls to the action server's calls, so the three runners count
    the same thing, every attempted call of the diagnostics server, a failed call included, as the state file counts a refused
    call. A reasoning step of Plan-Execute that names the server and no tool is no call."""
    from agent.plan_execute.models import StepResult
    from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, RIG_SERVER

    failed = StepResult(step_number=1, task="rank", server=CAUSAL_SERVER, response="", error="boom", tool="rank_root_causes")
    good = StepResult(step_number=2, task="rank", server=CAUSAL_SERVER, response="{}", tool="rank_root_causes")
    read = StepResult(step_number=3, task="read", server=RIG_SERVER, response="{}", tool="read_sensors")
    thought = StepResult(step_number=4, task="think", server=CAUSAL_SERVER, response="the pump", tool="none")
    plan = SimpleNamespace(raw="the plan", steps=[1, 2, 3, 4])
    pe = scripted_wiring_check.record_plan_execute(tmp_path / "pe", SimpleNamespace(plan=plan, answer="done", trajectory=[failed, good, read, thought]), [])
    records = [{"step": 1, "turn": 1, "agent": "CausalDiagnosisAgent", "task": "rank", "server": CAUSAL_SERVER, "tool": "rank_root_causes", "tool_args": {},
                "response": "boom", "error": "boom", "duration_ms": 1},
               {"step": 2, "turn": 2, "agent": "CausalDiagnosisAgent", "task": "rank", "server": CAUSAL_SERVER, "tool": "rank_root_causes", "tool_args": {},
                "response": "{}", "error": None, "duration_ms": 1},
               {"step": 3, "turn": 3, "agent": "IoTAgent", "task": "read", "server": RIG_SERVER, "tool": "read_sensors", "tool_args": {},
                "response": "{}", "error": None, "duration_ms": 1}]
    port = SimpleNamespace(answer="done", react_steps=4, react_trajectory={"task": "q"}, sub_agents=[])
    aat = agent_runs.record_agent_as_tool(tmp_path / "aat", port, [], records)
    trajectory = [{"server": CAUSAL_SERVER, "tool": "rank_root_causes", "error": "boom"}, {"server": CAUSAL_SERVER, "tool": "rank_root_causes", "error": None},
                  {"server": RIG_SERVER, "tool": "read_sensors", "error": None}]
    result = SimpleNamespace(trajectory=trajectory, plans=[], surprises=[], revisions=0, fallbacks=0, mode_of_first_plan="act_now", answer="done",
                             differential=[], model_calls=1, end_reason="finish", error=None, board={}, server=True)
    causal = run_episodes.record_causal(tmp_path / "causal", result, [])
    assert read_json(tmp_path / "causal" / "plans.json")["server"] is True
    assert pe["causal_calls"] == aat["causal_calls"] == causal["causal_calls"] == 2 and aat["react_steps"] == 4
    assert run_episodes.diagnostics_calls([(CAUSAL_SERVER, "rank_root_causes"), (CAUSAL_SERVER, None), (CAUSAL_SERVER, ""), (RIG_SERVER, "read_sensors")]) == 1


def test_a_model_agent_on_a_row_preset_plays_assetopsbench_servers_unless_asked_and_the_rest_keeps_the_rig():
    """Every row of the environment table, both arms of its rows and the healthy control
    run on AssetOpsBench's servers, so a model agent there defaults to --servers aob, while a reference policy, the scripted plan, the
    static agent and a development preset keep the rig's action server, and the option names a surface when given."""
    from CausalDiagnosis.experiments.episode_runners.run_configuration import ROW_PRESETS

    assert set(ROW_PRESETS) == {"base", "priced", "real_time", "advanced_real_time", "repair_rules", "deterioration", "deterioration_thinking_off",
                                "deterioration_thinking_price_double", "deterioration_thinking_price_half",
                                "deterioration_durations_off", "deterioration_mode_none", "deterioration_failure_off",
                                "deterioration_misdiagnosis_budget", "scheduled",
                                "healthy_control", "real_time_token_clock", "advanced_real_time_token_clock", "repair_rules_token_clock",
                                "deterioration_token_clock"}
    for agent in ("plan_execute", "agent_as_tool", "causal"):
        assert run_episodes.default_servers(agent, "real_time") == run_episodes.default_servers(agent, "scheduled") == "aob"
        assert run_episodes.default_servers(agent, "healthy_control") == run_episodes.default_servers(agent, "deterioration_token_clock") == "aob"
        assert run_episodes.default_servers(agent, "static") == run_episodes.default_servers(agent, "anchored") == "rig"
    for agent in ("oracle", "do_nothing", "greedy", "declare", "scripted", "static"):
        assert run_episodes.default_servers(agent, "real_time") == "rig"
    parser = run_episodes.build_parser()
    assert parser.parse_args(["--config", "real_time", "--agent", "causal"]).servers is None
    assert parser.parse_args(["--config", "real_time", "--agent", "causal", "--servers", "rig"]).servers == "rig"


# ----- the start check, the stamp, the guard of the test set and the cap of the development starts -----

def _parsed(*argv):
    a = run_episodes.build_parser().parse_args(list(argv))
    name, config, _prices = run_episodes.configure(a)
    return a, name, config


NO_REPOSITORY = "the tree is no git repository at its own root, as an upload of CausalDiagnosis/cluster_jobs/0_setup/upload_fork.sh is not"


def _no_repository(monkeypatch, tmp_path):
    """A tree that is no repository at its own root, whatever the tree under test is: rule 1 of the start check fails first."""
    monkeypatch.setattr(run_episodes.ledger.code_identity, "git_state", lambda paths, root=None: (None, None))


@pytest.mark.real_start_check
def test_a_model_agent_on_a_test_combination_is_refused_in_a_tree_that_is_no_repository(tmp_path, monkeypatch):
    """No model agent plays a test combination outside a run of record. In a tree that is no repository the start check gives its
    first reason, and main() stops before any run folder for every model agent, with the start, the agent and the reason."""
    _no_repository(monkeypatch, tmp_path)
    for agent in run_episodes.MODEL_AGENTS:
        out = tmp_path / agent
        with pytest.raises(SystemExit, match=f"--agent {agent} would play the test combination .* the tree is no git repository"):
            run_episodes.main(["--config", "static", "--agent", agent, "--starts", "combinations", "--seeds", "1", "--out", str(out)])
        assert not out.exists()


@pytest.mark.real_start_check
def test_the_guard_also_stops_an_episode_that_another_caller_starts(tmp_path, monkeypatch):
    """The loops of GEPA and every other caller build an EpisodeRunner themselves, so EpisodeRunner.start meets the guard on the
    start combination of the episode before it writes a launcher."""
    _no_repository(monkeypatch, tmp_path)
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world

    facts, cycles, blocks = load_world()
    _a, name, config = _parsed("--config", "static", "--agent", "plan_execute")
    runner = run_episodes.EpisodeRunner(config, name, "plan_execute", run_episodes.registered_servers("rig", config.diagnostics_server), cycles, blocks,
                                        facts, "guarded", {"judge": None, "grade": False, "ranking_model": run_episodes.DEFAULT_RANKING_MODEL})
    seed = start_states.combination_seeds(run_episodes.HydraulicRigEnv(cycles, blocks, config), 1)[0]
    with pytest.raises(SystemExit, match="--agent plan_execute would play the test combination"):
        runner.start(tmp_path / "ep", seed, config)
    assert not list((tmp_path / "ep").glob("*_server.py"))


def test_the_seam_of_the_conftest_takes_the_test_set_from_the_guard_and_leaves_the_start_check(tmp_path, monkeypatch):
    """Outside the mark the guard of a test holds no test combination, while the start check runs as it is, so a run of a test
    is stamped as every run of a tree that is no repository."""
    _no_repository(monkeypatch, tmp_path)
    a, name, config = _parsed("--config", "static", "--agent", "plan_execute")
    assert run_episodes.start_check.guarded_starts(config, None) == set()
    assert run_episodes.start_check.registration(a, config, name, default_facts_path())[0] == NO_REPOSITORY


@pytest.mark.real_start_check
def test_the_seam_of_the_conftest_is_off_under_the_mark_real_start_check():
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world

    _facts, _cycles, blocks = load_world()
    _a, _name, config = _parsed("--config", "static", "--agent", "plan_execute")
    assert run_episodes.start_check.guarded_starts(config, blocks)  # the test combinations of the asset, as the guard meets them


@pytest.mark.real_start_check
def test_a_reference_policy_and_the_scripted_check_run_stamped_on_test_combinations(tmp_path, monkeypatch):
    _no_repository(monkeypatch, tmp_path)
    for agent in ("oracle", "scripted"):
        run_dir = run_episodes.main(["--config", "static", "--agent", agent, "--seeds", "1", "--out", str(tmp_path / agent)])
        meta = read_json(run_dir / "run.json")
        assert meta["stamp"] == "development" and meta["stamp_reasons"][0] == NO_REPOSITORY
        assert meta["asset_id"] == "hyd_1" and meta["code"]["identity"] and meta["prices_sha256"] and set(meta["platform"]) >= {"python", "installed_sha256", "lock_sha256"}


@pytest.mark.real_start_check
def test_a_development_run_is_stamped_and_names_its_reasons(tmp_path, monkeypatch):
    _no_repository(monkeypatch, tmp_path)
    a, name, config = _parsed("--config", "base", "--agent", "causal", "--starts", "development")
    reasons = run_episodes.start_check.registration(a, config, name, default_facts_path())
    assert reasons[0] == NO_REPOSITORY
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world

    _facts, cycles, blocks = load_world()
    developed = run_episodes.development_config(config, blocks)
    starts = {tuple(c) for c in developed.allowed}
    run_episodes.guard_test_set("causal", starts, run_episodes.start_check.guarded_starts(config, blocks), reasons)  # development starts pass


@pytest.mark.real_start_check
def test_the_driver_and_the_wrappers_refuse_a_test_start_by_one_rule(tmp_path, monkeypatch):
    """The guard of the driver and the Gymnasium wrapper hold the same test set (development_split.held_starts) and refuse its
    first combination with the same sentence and the same reasons (development_split.guard_starts)."""
    from dataclasses import replace

    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import HydraulicRigGym

    _no_repository(monkeypatch, tmp_path)
    a, name, config = _parsed("--config", "base", "--agent", "causal")
    reasons = run_episodes.start_check.registration(a, config, name, default_facts_path())
    _facts, _cycles, blocks = load_world()
    held = run_episodes.start_check.guarded_starts(config, blocks)
    first = min(held)
    with pytest.raises(SystemExit) as driver:
        run_episodes.guard_test_set("causal", [first], held, reasons)
    fixed = replace(config, start={"combination": list(first), "entry_offset": 0})
    with pytest.raises(ValueError) as wrapper:
        HydraulicRigGym(fixed, default_facts_path(), starts="test", unregistered=reasons)
    assert reasons and str(driver.value).split(" would play ", 1)[1] == str(wrapper.value).split(" would play ", 1)[1]


@pytest.mark.real_start_check
def test_a_run_whose_rules_all_pass_is_of_record_and_one_changed_file_stamps_it(monkeypatch):
    """With the tree, the loaded files, the data and the open values all passing, a run of a table preset on a registered start set
    is of record; a path that differs from the commit is the reason that stamps it."""
    a, name, config = _parsed("--config", "base", "--agent", "oracle", "--starts", "sampled")
    facts = default_facts_path()
    for function in ("untracked_loaded", "data_reasons"):
        monkeypatch.setattr(run_episodes.ledger.code_identity, function, lambda *args, **kwargs: [])
    monkeypatch.setattr(run_episodes.ledger.code_identity, "open_reasons", lambda: [])
    monkeypatch.setattr(run_episodes.ledger.code_identity, "tree_record_reasons", lambda root=None: [])
    assert run_episodes.start_check.registration(a, config, name, facts) == []
    changed = "1 paths differ from the commit 0123456789ab, first CausalDiagnosis/experiments/episode_runners/run_episodes/__init__.py"
    monkeypatch.setattr(run_episodes.ledger.code_identity, "tree_record_reasons", lambda root=None: [changed])
    assert run_episodes.start_check.registration(a, config, name, facts) == [changed]


def test_a_registered_run_computes_no_table(tmp_path, monkeypatch):
    """A registered run opens the generated tables as they are, so the driver fits no model for it, while a run with a reason
    fits what its servers and its grade need, as before. A registered run plays on a clone of a commit (rule 3), so the tree is
    made one first (scratch_repository); the control: on no repository the registered run is refused."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit
    from CausalDiagnosis.experiments.run_verification.tests.test_run_ledger import scratch_repository
    from CausalDiagnosis.utils.atomic_writes_and_locks import READ_ONLY_TABLES

    monkeypatch.setenv(READ_ONLY_TABLES, "0")  # recorded here, so that teardown removes what the registered run sets
    fitted = []
    monkeypatch.setattr(fit, "ensure_models", lambda *args, **kwargs: fitted.append(args))
    monkeypatch.setattr(run_episodes.start_check, "registration", lambda *args, **kwargs: [])
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "no_repository"))
    with pytest.raises(SystemExit, match="which git does not track at the commit of the run"):
        run_episodes.main(["--config", "static", "--agent", "oracle", "--seeds", "1", "--out", str(tmp_path / "refused")])
    scratch_repository(monkeypatch, tmp_path)
    registered = run_episodes.main(["--config", "static", "--agent", "oracle", "--seeds", "1", "--out", str(tmp_path / "registered")])
    assert fitted == [] and "stamp" not in read_json(registered / "run.json")
    monkeypatch.setattr(run_episodes.start_check, "registration", lambda *args, **kwargs: ["a reason"])
    stamped = run_episodes.main(["--config", "static", "--agent", "oracle", "--seeds", "1", "--out", str(tmp_path / "stamped")])
    assert len(fitted) == 1 and read_json(stamped / "run.json")["stamp_reasons"] == ["a reason"]


def test_the_cap_draws_k_development_starts_that_cover_every_faulty_level_and_is_the_same_for_the_same_seed():
    healthy = (0, 0, 0)
    starts = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 0, 0)]
    capped = run_episodes.capped_starts(starts, 5, healthy, seed=0)
    assert len(capped) == 5 and capped == run_episodes.capped_starts(starts, 5, healthy, seed=0)
    assert [s for s in starts if s in capped] == capped  # in the order of the start set
    from CausalDiagnosis.datasets.splits.development_split import faulty_levels

    needed = set().union(*(faulty_levels(s, healthy) for s in starts))
    assert set().union(*(faulty_levels(s, healthy) for s in capped)) == needed
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world

    _facts, cycles, blocks = load_world()
    _a, _name, config = _parsed("--config", "base", "--agent", "causal")
    rig = run_episodes.development_config(config, blocks, 12)
    levels = [config.healthy[c] for c in config.components]
    assert len(rig.allowed) == 12 and set().union(*(faulty_levels(s, levels) for s in rig.allowed)) == set().union(
        *(faulty_levels(s, levels) for s in run_episodes.development_config(config, blocks).allowed))


def test_a_cap_below_what_the_coverage_needs_is_refused():
    with pytest.raises(SystemExit, match="--start-cap 1 is below the"):
        run_episodes.capped_starts([(1, 0), (0, 1)], 1, (0, 0))
    with pytest.raises(SystemExit, match="needs --starts development"):
        run_episodes.main(["--config", "static", "--agent", "oracle", "--start-cap", "3"])


def test_starts_development_plays_no_test_combination_on_any_registered_preset():
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world

    _facts, cycles, blocks = load_world()
    for preset in sorted(run_episodes.table_presets()):
        _a, _name, config = _parsed("--config", preset, "--agent", "causal")
        developed = run_episodes.development_config(config, blocks)
        assert not {tuple(c) for c in developed.allowed} & run_episodes.start_check.guarded_starts(config, blocks), preset


def test_seeds_below_one_are_refused_and_a_marker_run_still_plans_no_episode(tmp_path):
    for starts in run_episodes.SEED_STARTS:
        with pytest.raises(SystemExit, match="plans no episode"):
            run_episodes.main(["--config", "static", "--agent", "oracle", "--starts", starts, "--seeds", "0", "--out", str(tmp_path)])


def test_the_example_start_of_a_port_is_a_development_start_drawn_once_per_run():
    """The driver's record of the example start of the two paper-era ports (CausalDiagnosis.datasets.splits.development_split.example_start): a
    development start of the asset with the fewest faults, under the registered seed, as run.json records it."""
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
    from CausalDiagnosis.datasets.splits.development_split import development_starts, starts_of_test_set

    cycles = load_cycles(default_facts_path())
    blocks = build_blocks(cycles)
    _name, config = load_config("base")
    record = agent_runs.diagnosis_example(config, blocks)
    profile = get_profile(config.asset_id)
    levels = tuple(record["levels"][c] for c in profile.components)
    assert list(record["levels"]) == list(profile.components) and record["seed"] == 0
    assert levels in development_starts(profile, set(blocks)) and levels not in starts_of_test_set(profile, set(blocks))
    assert record["n_faulty"] == sum(level != HEALTHY[c] for c, level in zip(COMPONENTS, levels)) >= 1
    assert agent_runs.diagnosis_example(config, blocks) == record


def _launcher_expected(server, ranking_model, abstention, store, left_out, models_path, data_file):
    """The launcher text of a server and the options it reads from the private file of its folder: the paths of the machine,
    the data file, the models file and a store with a URL stand in launch_private.json, every other option in the text, in the
    order the driver gave them before launch_options had its own function."""
    from CausalDiagnosis.experiments.episode_runners import episode_launchers, run_configuration, run_folder_layout, servers_of_a_run

    folders = episode_launchers.asset_search_path()
    assets = ("import CausalDiagnosis.datasets  # noqa: E402\n"
              "CausalDiagnosis.datasets.__path__.extend(PRIVATE[\"asset_folders\"])  # the folders of the asset packages beside the package's own\n"
              ) if folders else ""
    if data_file is not None:
        assets += "import os  # noqa: E402\nos.environ.update(PRIVATE[\"environment\"])  # the data file of the run\n"
    options = []
    if server == servers_of_a_run.CAUSAL_SERVER:
        options.append(("ranking_model", ranking_model))
        if models_path is not None:
            options.append(("models_path", str(models_path)))
        options.append(("left_out", left_out))
    if server == servers_of_a_run.CAUSAL_SERVER and abstention and abstention[0] != episode_launchers.DEFAULT_ABSTENTION_RULE:
        options += [("abstention_rule", abstention[0]), ("abstention_cut", float(abstention[1]))]
    if server in servers_of_a_run.METERED_SERVERS:
        options += [("server", server), ("store", dict(store))]
    hidden = {key: value for key, value in options if key == "models_path" or (key == "store" and "url" in value)}
    extra = "".join(f", {key}=PRIVATE[\"options\"][{server!r}][{key!r}]" if key in hidden else f", {key}={value!r}"
                    for key, value in options)
    text = episode_launchers._LAUNCHER.format(server=server, module=episode_launchers.LAUNCHER_MODULES[server], private=episode_launchers.LAUNCH_PRIVATE, assets=assets,
                                   state=episode_launchers.STATE_FILES[server], extra=extra)
    return text, hidden


async def _listing_of_launched(path):
    """The hash of the tool listing that the process of a launcher file sends to a client over stdio."""
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from CausalDiagnosis.mcp_servers.utils import transcript

    # started as the driver starts a launcher, isolated (python -I), which the launcher's first lines require
    async with stdio_client(StdioServerParameters(command=sys.executable, args=["-I", str(path)], env=dict(os.environ))) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return transcript.listing_sha256((await session.list_tools()).tools)


def test_the_launchers_keep_their_bytes_and_launch_options_builds_the_same_server(tmp_path):
    """The launcher text of every server of both surfaces, with the default options and with a ranking model, an abstention rule, a
    models file, a left out combination, a data file and a store, equals the text the driver wrote before launch_options, and on an
    episode of each surface a server built in this process from launch_options lists the tools that the process of its launcher
    lists."""
    import asyncio
    import importlib

    import mcp.types as mcp_types

    from CausalDiagnosis.experiments.episode_runners import episode_launchers, run_configuration, run_folder_layout, servers_of_a_run
    from CausalDiagnosis.mcp_servers.utils import transcript
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_start_and_replay import start as rows_start
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import EpisodeStore
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
    from CausalDiagnosis.mcp_servers.utils.transcript_hooks import lowlevel_of

    rig, aob = servers_of_a_run.registered_servers("rig", True), servers_of_a_run.registered_servers("aob", True)
    couchdb = {"kind": "couchdb", "url": "http://couchdb:5984", "username": "admin", "password": "secret"}
    data_file = (run_configuration.get_profile(run_configuration.DEFAULT_ASSET).records["data_env"], tmp_path / "recordings.jsonl")
    cases = [(rig, {}),
             (rig, {"ranking_model": "discriminant_urn", "abstention": ("fixed_cut", 0.9), "left_out": (3, 100, 0, 130),
                    "models_path": tmp_path / "models.json", "data_file": data_file}),
             (aob, {"store": {"kind": "memory"}}), (aob, {"store": couchdb, "left_out": (3, 100, 0, 130), "data_file": data_file})]
    for number, (servers, options) in enumerate(cases):
        folder = tmp_path / f"case{number}"
        folder.mkdir()
        paths = episode_launchers.write_launchers(folder, servers, **options)
        private_options = {}
        for server in servers:
            expected, hidden = _launcher_expected(server, options.get("ranking_model", episode_launchers.DEFAULT_RANKING_MODEL), options.get("abstention"),
                                                  options.get("store"), options.get("left_out"), options.get("models_path"),
                                                  options.get("data_file"))
            assert paths[server].read_text(encoding="utf-8") == expected, (number, server)
            if hidden:
                private_options[server] = hidden
        private = json.loads((folder / episode_launchers.LAUNCH_PRIVATE).read_text(encoding="utf-8"))
        environment = {data_file[0]: str(data_file[1])} if "data_file" in options else {}
        assert private == {"src": [str(servers_of_a_run.ROOT), str(servers_of_a_run.SRC)], "asset_folders": episode_launchers.asset_search_path(), "environment": environment,
                           "options": private_options}, number
    _name, config = load_config("base")
    for servers, store in ((rig, None), (aob, {"kind": "memory"})):
        folder = tmp_path / ("aob" if store else "rig")
        folder.mkdir()
        (rows_start if store else episode.start)(folder / run_folder_layout.STATE_FILE, 0, config, default_facts_path())
        episode_store = EpisodeStore(folder / run_folder_layout.STATE_FILE, store) if store else None
        if episode_store:
            episode_store.create()
        try:
            paths = episode_launchers.write_launchers(folder, servers, store=store)
            for server in servers:
                options = episode_launchers.launch_options(server, store=store)
                app = lowlevel_of(importlib.import_module(episode_launchers.LAUNCHER_MODULES[server]).build_server(folder / episode_launchers.STATE_FILES[server], **options))
                listed = asyncio.run(app.request_handlers[mcp_types.ListToolsRequest](mcp_types.ListToolsRequest(method="tools/list")))
                assert transcript.listing_sha256(listed.root.tools) == asyncio.run(_listing_of_launched(paths[server])), server
        finally:
            if episode_store:
                episode_store.drop()


def test_the_settings_of_a_port_record_the_temperature_of_its_run():
    """run.json of a sampling run of a port records one temperature: the decoding of the requests and the limits of the port."""
    from CausalDiagnosis.experiments.episode_runners import episode_launchers, run_configuration, run_folder_layout, servers_of_a_run

    servers = servers_of_a_run.registered_servers("aob", False)
    for agent in ("plan_execute", "agent_as_tool"):
        for temperature, recorded in ((0.7, 0.7), (None, 0.0)):
            runner = SimpleNamespace(temperature=temperature, servers=servers)
            settings = run_episodes.agent_settings(SimpleNamespace(agent=agent), runner)
            assert settings["decoding"]["temperature"] == settings["limits"]["decoding"]["temperature"] == recorded, (agent, temperature)


def test_a_couchdb_store_without_its_password_stops_at_launch(monkeypatch):
    a = SimpleNamespace(servers="aob", store="couchdb")
    monkeypatch.setenv("COUCHDB_URL", "http://127.0.0.1:5984")
    monkeypatch.delenv("COUCHDB_PASSWORD", raising=False)
    with pytest.raises(SystemExit, match="COUCHDB_PASSWORD"):
        run_episodes.store_settings(a)
    monkeypatch.setenv("COUCHDB_PASSWORD", "drawn-per-job")
    assert run_episodes.store_settings(a)["password"] == "drawn-per-job"


def test_a_module_outside_every_list_stops_the_run_after_its_first_episode(tmp_path, monkeypatch):
    """The check of the loaded files runs after the first episode as well as at the end, so a module that lies in no list voids one
    episode, not a whole run."""
    monkeypatch.setattr(run_episodes.run_record, "loaded_outside_lists", lambda: ["CausalDiagnosis/unlisted.py"])
    with pytest.raises(SystemExit, match="unlisted.py"):
        run_episodes.main(["--config", "static", "--agent", "do_nothing", "--seeds", "2", "--no-grade", "--out", str(tmp_path / "out")])
    (run_dir,) = [p for p in (tmp_path / "out").iterdir() if p.is_dir()]
    assert not (run_dir / "run.json").exists() and len(_rows(run_dir)) == 1


def test_the_static_line_on_the_assetopsbench_rows_reads_their_first_cycle_and_drops_its_store(tmp_path, monkeypatch):
    """On the rows of AssetOpsBench the static answer sees the cycle that the first read of that surface serves, the latest reading
    of the full group of a RecordedRigEnv, and the episode exports and drops its databases as a played episode does."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_documents import EXPORT_NAME
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import STORE_NAME

    server, _requests = _stand_in_endpoint(fixed_reply=STATIC_REPLY)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        run_dir, rows, meta = _run(["--config", "static", "--agent", "static", "--model", "vendor/stand-in-1.0", "--base-url", base, "--seeds", "1",
                                    "--no-grade", "--servers", "aob", "--store", "memory"], tmp_path / "static_aob")
    finally:
        server.shutdown()
    assert meta["mode"] == "static" and len(rows) == 1 and rows[0]["error"] == ""
    ep_dir = run_dir / f"seed{int(rows[0]['seed']):06d}"
    assert not (ep_dir / STORE_NAME).exists() and (ep_dir / EXPORT_NAME).exists()
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    _facts, cycles, blocks = load_world()
    config = EpisodeConfig.from_json(episode.load_state(ep_dir / run_episodes.STATE_FILE)["config"])
    env = RecordedRigEnv(cycles, blocks, config)
    env.reset(int(rows[0]["seed"]))
    env.step(run_episodes.first_read(env, run_episodes.full_sensor_group(config)))
    first = next(e for e in reversed(env.log) if e["event"] == "read")["served"][0]["cycle"]
    assert read_json(ep_dir / HIDDEN / "static.json")["cycle_index"] == first


def test_a_registered_run_holds_the_tables_for_itself_and_the_servers_it_starts(monkeypatch):
    """A run whose start check gives no reason sets CD_TABLES_READ_ONLY=1, which every server process it starts inherits, and a
    run with a reason leaves it unset."""
    import os

    from CausalDiagnosis.utils.atomic_writes_and_locks import READ_ONLY_TABLES

    monkeypatch.setenv(READ_ONLY_TABLES, "0")  # recorded here, so that teardown removes what hold_tables sets
    monkeypatch.delenv(READ_ONLY_TABLES, raising=False)
    run_episodes.hold_tables(["the preset is no registered start set"])
    assert READ_ONLY_TABLES not in os.environ
    run_episodes.hold_tables([])
    assert os.environ[READ_ONLY_TABLES] == "1"
