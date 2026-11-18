"""The episode driver plays any registered asset: two made-up ports, copied to pytest's temporary folder and mounted from there,
play the six rows, the three agents without and with the CausalDiagnosis MCP, through run_episodes/ by their asset id, every
server of a row answers for the port and none for hyd_1, and every part of the drivers that reads an asset reads the asset of
the run or refuses another one by name. demo_1 of the asset layer tests is of an evidence kind of its own and its recordings
admit no development split, so on it the CausalDiagnosis agent without the MCP plays and the five other rows stop with their
reason. bench_1, beside this file, is of the recorded levels kind and its recordings admit a split, so every row plays on it.
The agents talk to scripted stand-ins of their models. No network, no model, and no file of the tree is written."""
from CausalDiagnosis.utils.paths import CONFIGS, PACKAGE
import csv
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import read_rows
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import episode_launchers, run_configuration, servers_of_a_run
from CausalDiagnosis.utils import csv_rows  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import episode_judge as judge  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs import precompute_grid_tables as precompute_tables  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents import scripted_wiring_check  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_reference_policies as run_policies  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.datasets.tests.conftest import PORTS as LAYER_PORTS, mount_port, write_port_records, write_recordings  # noqa: E402
from CausalDiagnosis.datasets.splits.development_split import development_starts  # noqa: E402
from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.utils import model_failures
from CausalDiagnosis.mcp_servers.utils import transcript  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit, diagnosability_grade as grade  # noqa: E402
from CausalDiagnosis.utils.public_rules.visits import is_no_call  # noqa: E402

from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests.scripted_endpoint import REVIEW_OK, ScriptedEndpoint, finish, step  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper.tests.test_plan_execute_paper import VALID, plan  # noqa: E402

from CausalDiagnosis.utils import model_request_retry as retry  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import react_tool_adapter  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import ORIENTATION_HEADING, SUMMARY_HEADING  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.tests.test_episode_judge import RUBRIC_PASS, FakeJudge, episode_folder  # noqa: E402

DEMO, BENCH = "demo_1", "bench_1"
OUR_PORTS = Path(__file__).resolve().parent / "ports"  # bench_1, beside this file
MODEL = "granite-stand-in"
END_TASK = react_tool_adapter.agent_name("finish")  # the name under which the agents see the task tool finish
FINISH = {"kind": "finish", "args": {}, "expect": {}, "evidence": []}


def port_names(asset_id):
    """The names an answer about the asset can carry: its id, its site, its components and the registry asset of each."""
    profile = get_profile(asset_id)
    site = profile.text["fallback"]["site"]
    return [asset_id, site, site.upper(), *profile.components, *(f"{asset_id}_{c}" for c in profile.components)]


def rig_names():
    """The names of hyd_1 that no answer of another asset may carry, each matched as a whole word."""
    profile = get_profile(DEFAULT_ASSET)
    return [DEFAULT_ASSET, profile.text["fallback"]["site"], *profile.components]


@pytest.fixture(scope="module")
def mounted(tmp_path_factory):
    """Both made-up ports copied to a temporary folder and mounted from there, each with its records written and named by the
    variable of its asset file, for the length of the module, so that the models the driver fits and the tables a label
    computes land in that folder."""
    root = tmp_path_factory.mktemp("ports")
    for source in (LAYER_PORTS / DEMO, OUR_PORTS / BENCH):
        shutil.copytree(source, root / source.name, ignore=shutil.ignore_patterns("data", "__pycache__"))
    with mount_port(root), pytest.MonkeyPatch.context() as patch:
        patch.setenv(get_profile(DEMO).records["data_env"], str(write_port_records(root / "demo_records.jsonl")))
        patch.setenv(get_profile(BENCH).records["data_env"], str(write_recordings(root / "bench_cycles.jsonl", get_profile(BENCH))))
        yield root


def world(asset_id):
    _facts, cycles, blocks = run_configuration.load_world(None, asset_id)
    return cycles, blocks


# ----- the stand-ins of the models of the three agents, one call of every server of the row each -----

def paper_era_calls(asset_id, mcp):
    """(agent, tool, arguments) of one call of every server of a row on the AssetOpsBench servers, as the step agents of
    Plan-Execute and the sub-agents of Agent-As-Tool name them."""
    profile = get_profile(asset_id)
    site, component = profile.text["fallback"]["site"], profile.components[0]
    calls = [("TaskAgent", "get_asset_profile", "{}"), ("IoT", "assets", f"site_name={site}"),
             ("FMSR", "get_failure_modes", f"asset_class={component.upper()}"), ("WO", "list_workorders", f"site_id={site}")]
    return calls + ([("CausalDiagnosisAgent", "rank_root_causes", "top_k=3")] if mcp else [])


PLAN_EXECUTE_AGENTS = {"TaskAgent": "TaskAgent", "IoT": "IoT Data Download", "FMSR": "Failure Mode and Sensor Relevancy Expert for Industrial Asset",
                       "WO": "Work Order Management", "CausalDiagnosisAgent": "CausalDiagnosisAgent"}
AGENT_AS_TOOL_AGENTS = {"TaskAgent": "TaskAgent", "IoT": "IoTAgent", "FMSR": "FMSRAgent", "WO": "WOAgent", "CausalDiagnosisAgent": "CausalDiagnosisAgent"}


def plan_execute_script(asset_id, mcp, calls=None):
    calls = paper_era_calls(asset_id, mcp) if calls is None else calls
    steps = [(f"Call {tool}.", PLAN_EXECUTE_AGENTS[agent], "None") for agent, tool, _ in calls] + [("End the task.", "TaskAgent", "None")]
    script = [plan(*steps), VALID]
    for _agent, tool, arguments in calls:
        script += [*step(f"I need {tool}.", tool, arguments), *finish(f"{tool} answered"), REVIEW_OK]
    return script + step("I need to end the task.", END_TASK, "summary=no repair was made")


def agent_as_tool_script(asset_id, mcp, calls=None):
    script = []
    for agent, tool, arguments in paper_era_calls(asset_id, mcp) if calls is None else calls:
        script += [*step(f"I need the {agent}.", AGENT_AS_TOOL_AGENTS[agent], f"request=call {tool}"),
                   *step(f"I need {tool}.", tool, arguments), *finish(f"{tool} answered"), REVIEW_OK]
    return script + [*step("I should end the task.", "TaskAgent", "request=end the task with the summary no repair was made"),
                     *step("I need to end the task.", END_TASK, "summary=no repair was made")]


# the board lines of the causal agent's planning turn that ask for the opening reads, which the planner answers as a model would
NO_PROFILE = "No profile read yet"
STILL_TO_READ = re.compile(r"Still to read before a reading on this row, as steps of their own: ([a-z_, ]+)\.")


def plan_reply(steps, mode="act_now"):
    return json.dumps({"branch": 1, "mode": mode, "why": "scripted: the row is played through once", "steps": steps})


def causal_reply(asset_id):
    """The planner of the causal agent without a model: an orientation that names two components of the asset, the opening
    plan that the controller asks for, the reads the surface asks for before a reading, then a plan that finishes."""
    differential = [{"component": c, "expected_symptoms": "a shifted reading", "failure_mode_ids": []} for c in get_profile(asset_id).components[:2]]

    def reply(prompt):
        if ORIENTATION_HEADING in prompt:
            return json.dumps({"differential": differential})
        if SUMMARY_HEADING in prompt:
            return "Scripted causal loop complete."
        if NO_PROFILE in prompt:
            return plan_reply([{"kind": "profile"}, {"kind": "failure_modes", "args": {"component": ""}}, {"kind": "fix_history"}], "test_first")
        missing = STILL_TO_READ.search(prompt)
        if missing:
            return plan_reply([{"kind": kind.strip()} for kind in missing.group(1).split(",")], "test_first")
        return plan_reply([dict(FINISH)])
    return reply


def chat_endpoint(reply):
    """A local OpenAI compatible chat endpoint that answers the last message of every request with reply(text)."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import answer_context_route

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the http.server hook name
            if not answer_context_route(self, None):
                self.send_error(404)

        def do_POST(self):  # noqa: N802 - the http.server hook name
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if answer_context_route(self, body):
                return
            answer = {"id": "planner", "object": "chat.completion", "created": int(time.time()), "model": body["model"],
                      "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply(body["messages"][-1]["content"])}}],
                      "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
            data = json.dumps(answer).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def play(agent, args, monkeypatch, asset_id, mcp):
    """run_episodes.main on args with the stand-in of the model of the agent, the run folder it returns."""
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    if agent == design.OURS:
        server = chat_endpoint(causal_reply(asset_id))
        try:
            return run_episodes.main([*args, "--model", "vendor/stand-in-1.0", "--base-url", f"http://127.0.0.1:{server.server_address[1]}/v1/"])
        finally:
            server.shutdown()
    script = (plan_execute_script if agent == design.PLAN_EXECUTE else agent_as_tool_script)(asset_id, mcp)
    with ScriptedEndpoint(script, model=MODEL) as endpoint:
        return run_episodes.main([*args, "--model", MODEL, "--base-url", endpoint.base_url])


def refusal(asset_id, agent, mcp):
    """A pattern of the message with which a row of the asset stops at launch, or None for a row that plays. For a row with the MCP
    it is the answer of the one place where the driver decides how such a row gets its models (run_episodes.diagnostics_models,
    which test_a_row_with_the_mcp_reads_the_models_of_its_asset pins), and for a paper-era design the refusal of an asset whose
    recordings admit no development split, from which the example start of its prompt is drawn."""
    facts, _cycles, blocks = run_configuration.load_world(None, asset_id)
    if mcp:
        try:
            run_episodes.diagnostics_models(asset_id, (run_episodes.CAUSAL_SERVER,), blocks, facts, fit_models=False)
        except SystemExit as stop:
            return re.escape(str(stop))
    if agent in (design.PLAN_EXECUTE, design.AGENT_AS_TOOL):
        try:
            development_starts(get_profile(asset_id), set(blocks))
        except ValueError:
            return "diagnosis example"
    return None


def answers(ep_dir):
    """{server: [text]} of every call of the agent in the transcript of an episode, the text that the server kept or the
    response that the runner recorded under the same hash. A no_call step, which the driver takes for an iteration without a
    counted call, is no call of the agent and no record of the runner, so its line is left out (visits.is_no_call)."""
    recorded = {transcript.text_sha256(s["response"]): s["response"] for s in read_json(ep_dir / "trajectory.json") if s.get("tool")}
    found = {}
    for line in transcript.read(ep_dir):
        if line["kind"] == "call" and not is_no_call(line):
            found.setdefault(line["server"], []).append(line.get("answer") or recorded[line["answer_sha256"]])
    return found


@pytest.mark.parametrize("asset_id", [DEMO, BENCH])
@pytest.mark.parametrize("row", design.AGENT_ROWS, ids=lambda row: row[2])
def test_the_six_rows_play_the_made_up_ports_or_stop_by_reason(mounted, tmp_path, monkeypatch, asset_id, row):
    """Each of the six rows, the three agents without and with the CausalDiagnosis MCP, through run_episodes.main with --asset
    and one start. A row that cannot play the asset stops before any run folder exists, by its reason. A row that plays finishes
    its episode without an error, run.json and its data record name the asset and its records, every launcher finds the mounted
    port, and every server of the run answers at least once with a name of the port and never with a name of hyd_1."""
    agent, mcp, _label = row
    out = tmp_path / "runs"
    args = ["--config", "base", "--asset", asset_id, "--agent", agent, "--seeds", "1", "--store", "memory", "--out", str(out),
            *([] if mcp else ["--no-diagnostics"])]
    expected = refusal(asset_id, agent, mcp)
    if expected:
        with pytest.raises(SystemExit, match=expected):
            play(agent, args, monkeypatch, asset_id, mcp)
        assert not out.exists() or not any(out.iterdir())
        return
    run_dir = play(agent, args, monkeypatch, asset_id, mcp)
    rows, meta = read_rows(run_dir), read_json(run_dir / "run.json")
    assert len(rows) == 1 and rows[0]["error"] == "", rows[0]
    assert meta["asset_id"] == meta["config"]["asset_id"] == asset_id
    facts = Path(run_configuration.records.default_facts_path(get_profile(asset_id)))
    assert meta["data"]["facts"] == ledger.sha256_of(facts)
    assert not {"labels", "grades"} & set(meta["data"]), meta["data"]  # the tables of hyd_1 belong to no run of another asset
    ep_dir = run_dir / "seed000000"
    assert str(mounted) in json.loads((ep_dir / episode_launchers.LAUNCH_PRIVATE).read_text(encoding="utf-8"))["asset_folders"]
    for launcher in ep_dir.glob("*_server.py"):
        assert str(mounted) not in launcher.read_text(encoding="utf-8"), launcher.name  # the launcher names no path of the machine
    by_server = answers(ep_dir)
    names, rig = port_names(asset_id), [re.compile(rf"\b{re.escape(n)}\b") for n in rig_names()]
    for server in meta["servers"]:
        texts = by_server.get(server, [])
        assert any(name in text for text in texts for name in names), (server, texts[:2])
        assert not [p.pattern for p in rig for text in texts if p.search(text)], server


def play_failing_first(agent, tmp_path, monkeypatch, held=False):
    """One short episode of a port on bench_1 without the MCP (one call of a server, then the end of the task) against an
    endpoint that fails the first attempt of every POST, with the registered wait set to zero: (run folder, the endpoint's
    stats). held opens one request beside the port's for the whole episode, headers sent and body held back, as a port that
    opened a second request would."""
    monkeypatch.setattr(retry, "_wait", lambda deadline, delay_s: 0.0)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    calls = paper_era_calls(BENCH, False)[:1]
    script = (plan_execute_script if agent == design.PLAN_EXECUTE else agent_as_tool_script)(BENCH, False, calls)
    args = ["--config", "base", "--asset", BENCH, "--agent", agent, "--seeds", "1", "--store", "memory",
            "--out", str(tmp_path / "runs"), "--no-diagnostics"]
    with ScriptedEndpoint(script, model=MODEL, fail_first=True) as endpoint:
        extra = None
        if held:
            extra = socket.create_connection(endpoint.server.server_address)
            extra.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n")
            deadline = time.monotonic() + 10
            while endpoint.stats["open"] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert endpoint.stats["open"] == 1
        try:
            run_dir = run_episodes.main([*args, "--model", MODEL, "--base-url", endpoint.base_url])
        finally:
            if extra is not None:
                extra.sendall(b"{}")
                extra.recv(65536)
                extra.close()
        return run_dir, endpoint.stats


def check_one_request_at_a_time(run_dir, stats):
    """No request opens while another is open, the retry of every failed attempt is the next
    request to arrive, and the episode's ledger holds one span per failed attempt, read back by the servers' reader."""
    assert stats["failed"] > 0, stats
    assert stats["most_open"] == 1, stats["most_open"]
    arrivals = stats["arrivals"]
    for key in set(arrivals):
        first = arrivals.index(key)
        assert first + 1 < len(arrivals) and arrivals[first + 1] == key, (first, key[:200])
    ep_dir = run_dir / "seed000000"
    lines = (ep_dir / model_failures.NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == stats["failed"], (len(lines), stats["failed"])
    entries = [json.loads(line) for line in lines]
    assert all(e["status"] == 503 and e["next_start"] >= e["start"] for e in entries), entries
    spans, _cursor = model_failures.failed_spans_since(ep_dir, 0)
    assert len(spans) == len(entries)


@pytest.mark.parametrize("agent", [design.PLAN_EXECUTE, design.AGENT_AS_TOOL])
def test_a_port_opens_one_model_request_at_a_time_and_records_each_failed_attempt(mounted, tmp_path, monkeypatch, agent):
    """Each port plays one short episode against an endpoint that fails the first attempt of every request: the episode ends
    without an error, the port never has two requests open, sends each failed request again as its next request, and
    model_failures.jsonl holds one line per failed attempt."""
    run_dir, stats = play_failing_first(agent, tmp_path, monkeypatch)
    rows = read_rows(run_dir)
    assert len(rows) == 1 and rows[0]["error"] == "", rows[0]
    check_one_request_at_a_time(run_dir, stats)


@pytest.mark.parametrize("agent", [design.PLAN_EXECUTE, design.AGENT_AS_TOOL])
def test_the_check_fails_when_a_second_request_is_open(mounted, tmp_path, monkeypatch, agent):
    """The control of the test above: with one more request held open through the episode, as a port that opened a second
    request would leave it, the endpoint counts two requests open at once and the check fails on most_open."""
    run_dir, stats = play_failing_first(agent, tmp_path, monkeypatch, held=True)
    assert stats["most_open"] == 2, stats["most_open"]
    with pytest.raises(AssertionError, match=r"^2\b"):
        check_one_request_at_a_time(run_dir, stats)


def test_the_asset_comes_from_the_preset_or_the_option(mounted, tmp_path):
    """--asset composes a preset over the defaults of the asset it names, a preset that names another asset stops by name, an id
    without a folder stops with the ids found, and without --asset a preset composes over the defaults of the engine as before."""
    name, config = run_configuration.load_config("base", BENCH)
    assert (name, config.asset_id) == ("base", BENCH)
    assert config.sensor_groups == get_profile(BENCH).default_config().sensor_groups
    assert run_configuration.load_config("base")[1] == run_configuration.load_config("base", DEFAULT_ASSET)[1]
    assert run_configuration.load_config("base")[1].asset_id == DEFAULT_ASSET
    named = tmp_path / "named_rig.json"
    named.write_text(json.dumps({**json.loads((CONFIGS / "layer_presets" / "base.json").read_text(encoding="utf-8")), "asset_id": DEFAULT_ASSET}),
                     encoding="utf-8")
    with pytest.raises(SystemExit, match=f"the preset names the asset {DEFAULT_ASSET}, and --asset names {BENCH}"):
        run_configuration.load_config(str(named), BENCH)
    with pytest.raises(SystemExit, match=f"names no registered asset; the asset folders found are .*{BENCH}"):
        run_configuration.load_config("base", "no_such_1")


def test_every_server_process_finds_a_port_mounted_beside_the_package(mounted, tmp_path, monkeypatch):
    """Every launcher puts the folder of the mounted port on the search path of its process and points the data variable of the
    asset at the data file of the run, so that a process started from the launcher's own lines, which inherits neither, finds the
    profile and the data file; the diagnostics launcher names a models file only when the driver gives one."""
    profile = get_profile(BENCH)
    variable, facts = profile.records["data_env"], Path(run_configuration.records.default_facts_path(profile))
    servers = (run_episodes.RIG_SERVER, run_episodes.CAUSAL_SERVER)
    (tmp_path / "with").mkdir()
    (tmp_path / "without").mkdir()
    launchers = episode_launchers.write_launchers(tmp_path / "with", servers, models_path=fit.models_path(BENCH), data_file=(variable, facts))
    private = json.loads((tmp_path / "with" / episode_launchers.LAUNCH_PRIVATE).read_text(encoding="utf-8"))
    assert str(mounted) in private["asset_folders"] and private["environment"] == {variable: str(facts)}
    for path in launchers.values():
        text = path.read_text(encoding="utf-8")
        assert str(mounted) not in text and str(facts) not in text and "CausalDiagnosis.datasets.__path__.extend" in text, path.name
    head = launchers[run_episodes.RIG_SERVER].read_text(encoding="utf-8").split("\nfrom CausalDiagnosis.mcp_servers.")[0]
    probe = f"\nfrom CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile\nprint(get_profile({BENCH!r}).asset_id, os.environ[{variable!r}])\n"
    alone = {k: v for k, v in os.environ.items() if k != variable}
    (tmp_path / "with" / "probe_server.py").write_text(head + probe, encoding="utf-8")  # beside the private file, as a launcher reads it
    # started as the driver starts a launcher, isolated (python -I), which the launcher's first lines require
    started = subprocess.run([sys.executable, "-I", str(tmp_path / "with" / "probe_server.py")], capture_output=True, text=True, env=alone,
                             timeout=120)
    assert started.returncode == 0 and started.stdout.split() == [BENCH, str(facts)], started.stderr[-800:]
    bare = f"import os, sys\nsys.path[:0] = {[str(servers_of_a_run.ROOT), str(servers_of_a_run.SRC)]!r}" + probe
    assert subprocess.run([sys.executable, "-c", bare], capture_output=True, text=True, env=alone, timeout=120).returncode != 0
    assert private["options"][run_episodes.CAUSAL_SERVER]["models_path"] == str(fit.models_path(BENCH))
    assert "models_path=PRIVATE" in launchers[run_episodes.CAUSAL_SERVER].read_text(encoding="utf-8")
    monkeypatch.setattr(episode_launchers, "asset_search_path", lambda: [])
    plain = episode_launchers.write_launchers(tmp_path / "without", servers)
    for path in plain.values():
        text = path.read_text(encoding="utf-8")
        assert "CausalDiagnosis.datasets" not in text and "os.environ" not in text and "models_path" not in text, path.name


def test_a_row_with_the_mcp_reads_the_models_of_its_asset(mounted):
    """The one place where the driver decides how a row with the diagnostics server gets its models: each asset has its own models
    file, a row with the server on an asset whose models the model code cannot fit stops by name and names the row without the
    server, and on an asset it can fit the models are fitted on the data file of the run before any server starts."""
    assert fit.models_path(DEFAULT_ASSET) == fit.MODELS_PATH
    assert fit.models_path(BENCH) == mounted / BENCH / "data" / "models.json"
    demo_facts, _cycles, demo_blocks = run_configuration.load_world(None, DEMO)
    bench_facts, _cycles, bench_blocks = run_configuration.load_world(None, BENCH)
    assert fit.unfit_reasons(BENCH, set(bench_blocks)) == []
    reasons = fit.unfit_reasons(DEMO, set(demo_blocks))
    assert len(reasons) == 3 and "demo_history" in reasons[0] and "value" in reasons[1] and "development split" in reasons[2]
    with pytest.raises(SystemExit, match=rf"fits none for the asset {DEMO}: .*Play the row without the MCP \(--no-diagnostics\)"):
        run_episodes.diagnostics_models(DEMO, (run_episodes.RIG_SERVER, run_episodes.CAUSAL_SERVER), demo_blocks, demo_facts, fit_models=True)
    assert run_episodes.diagnostics_models(DEMO, (run_episodes.RIG_SERVER,), demo_blocks, demo_facts, fit_models=True) is None
    assert not fit.models_path(DEMO).exists()
    path = run_episodes.diagnostics_models(BENCH, (run_episodes.RIG_SERVER, run_episodes.CAUSAL_SERVER), bench_blocks, bench_facts,
                                           fit_models=True)
    assert path == fit.models_path(BENCH) and read_json(path)["facts_sha256"] == fit.facts_sha256(bench_facts)
    assert run_episodes.diagnostics_models(BENCH, (run_episodes.RIG_SERVER,), bench_blocks, bench_facts, fit_models=False) is None


def test_the_label_of_a_start_is_the_declaration_of_its_asset(mounted):
    """The label of a start is the function that LABELS of its asset names, on the rig the label of diagnosability_label.py, and an asset
    that declares none gets no label, under which the verifier counts every start separable."""
    assert ports.declared(DEFAULT_ASSET, "LABELS") == precompute_tables.SEPARABILITY_LABEL
    bench = SimpleNamespace(config=SimpleNamespace(asset_id=BENCH))
    assert run_episodes.grade_of_start(bench, {"rules": {}}) == (None, "no label declared by the asset")
    profile = get_profile(DEMO)
    healthy = SimpleNamespace(config=SimpleNamespace(asset_id=DEMO), initial=profile.healthy_state(), profile=profile)
    label, source = run_episodes.grade_of_start(healthy, {"rules": {}})
    assert label["separable"] is True and label["e18"]["separable"] is True and source == "table"


def test_the_static_agent_reads_the_full_sensor_group_of_its_asset(mounted):
    """The static answer reads the sensor group with the most channels of the asset of the run, and on a port the facts and the
    failure modes it is shown are those of the port."""
    groups = {asset_id: run_episodes.full_sensor_group(run_configuration.load_config("base", asset_id)[1]) for asset_id in (DEFAULT_ASSET, BENCH, DEMO)}
    assert groups == {DEFAULT_ASSET: "tier3", BENCH: "full", DEMO: "all"}
    cycles, blocks = world(BENCH)
    context = run_episodes.static_context(cycles, blocks, run_configuration.load_config("base", BENCH)[1], 0)
    assert context["features"] and {mode["component"] for mode in context["failure_modes"]} == set(get_profile(BENCH).components)


# the plans of the scripted wiring check on hyd_1 as literals, step by step, from before the plan followed the asset of the run
RIG, CD = run_episodes.RIG_SERVER, run_episodes.CAUSAL_SERVER
TASK, IOT, FMSR, WO = run_episodes.TASK_SERVER, run_episodes.IOT_SERVER, run_episodes.FMSR_SERVER, run_episodes.WO_SERVER
RIG_DIAGNOSIS = '{"cooler": 100, "valve": 100, "pump": 1, "accumulator": 130, "evidence_ids": ""}'
RIG_CHEAPEST_FIELDS = ["TS1_Temperature_C_1Hz", "TS2_Temperature_C_1Hz", "TS3_Temperature_C_1Hz", "TS4_Temperature_C_1Hz",
                       "VS1_Vibration_mm_per_s_1Hz", "CE_CoolingEfficiency_percent_1Hz"]
RIG_STEPS = [
    ("Read the asset profile", RIG, "get_asset_profile", "{}"),
    ("Read the failure modes of the pump", RIG, "get_failure_modes", '{"component": "pump"}'),
    ("Read the fix history of the rig", RIG, "get_fix_history", "{}"),
    ("Look up the repair manual of the pump", RIG, "lookup_fix", '{"component": "pump"}'),
    ("Read one cycle of the cheapest sensor group", RIG, "read_sensors", '{"sensor_group": "tier1", "cycles": 1}'),
    ("Read one cycle of the full sensor group", RIG, "read_sensors", '{"sensor_group": "tier3", "cycles": 1}'),
    ("List the released evidence", CD, "get_episode_evidence", '{"sensor_group": "all"}'),
    ("Rank the candidate fault combinations", CD, "rank_root_causes", '{"top_k": 3}'),
    ("Summarize the changes since the start", RIG, "summarize_changes_since", '{"cycle": 0}'),
    ("Submit a diagnosis", RIG, "submit_diagnosis", RIG_DIAGNOSIS),
    ("Predict the effect of replacing the pump", CD, "estimate_intervention_effect", '{"action": "replace:pump"}'),
    ("Create a work order for the pump", RIG, "create_work_order", '{"component": "pump", "prediction_id": "pred/1", "evidence_ids": ""}'),
    ("Wait until the technician starts the repair", RIG, "wait_for_notification", '{"timeout": 5}'),
    ("Wait until the work order closes", RIG, "wait_for_notification", '{"timeout": 5}'),
    ("Read the clock", RIG, "get_current_time", "{}"),
    ("Read one cycle of the cheapest sensor group after the repair", RIG, "read_sensors", '{"sensor_group": "tier1", "cycles": 1}'),
    ("Verify the prediction against the reading", CD, "verify_outcome", '{"prediction_id": "pred/1"}'),
    ("Try to declare the evidence insufficient after the diagnosis", RIG, "declare_insufficient_evidence",
     '{"sensor_group": "tier3", "cycles": 1, "evidence_ids": ""}'),
    ("End the episode", RIG, "finish", '{"summary": "scripted run complete"}'),
]
RIG_AOB_STEPS = [
    ("Read the asset profile", TASK, "get_asset_profile", "{}"),
    ("Read the asset registry of the site", IOT, "assets", '{"site_name": "MAIN"}'),
    ("Read the failure modes of the pump", FMSR, "get_failure_modes", '{"asset_class": "PUMP"}'),
    ("Read the recorded history of the cheapest sensor group", IOT, "history",
     json.dumps({"site_name": "MAIN", "asset_id": "hyd_1", "sensors": RIG_CHEAPEST_FIELDS, "limit": 5})),
    ("Read the newest recorded row of every field", IOT, "latest_reading", '{"site_name": "MAIN", "asset_id": "hyd_1"}'),
    ("List the released evidence", CD, "get_episode_evidence", '{"sensor_group": "all"}'),
    ("Rank the candidate fault combinations", CD, "rank_root_causes", '{"top_k": 3}'),
    ("Submit a diagnosis", TASK, "submit_diagnosis", RIG_DIAGNOSIS),
    ("Create a work order for the pump", WO, "generate_work_order",
     '{"description": "replace the pump", "asset_num": "hyd_1_pump", "site_id": "MAIN", "priority": 2, "work_type": "CM"}'),
    ("List the work orders of the site", WO, "list_workorders", '{"site_id": "MAIN"}'),
    ("Read the failure code catalog", WO, "get_failure_codes", "{}"),
    ("End the episode", TASK, "finish", '{"summary": "scripted run complete"}'),
]


def test_the_scripted_plan_of_the_rig_is_unchanged_and_follows_the_asset(mounted):
    """The rule that builds the scripted plan from the asset profile gives the rig its plan of before, step by step, and on a port
    a plan of the port's own names: the component whose work order takes longest to close, its first faulty level, the cheapest
    and the full sensor group, the site, the asset class of the component and its child asset, and no name of hyd_1."""
    rig = get_profile(DEFAULT_ASSET)
    assert scripted_wiring_check.ScriptedBackend.STEPS == scripted_wiring_check.scripted_steps(rig) == RIG_STEPS
    assert scripted_wiring_check.ScriptedBackend.AOB_STEPS == scripted_wiring_check.scripted_steps(rig, aob=True) == RIG_AOB_STEPS
    bench = get_profile(BENCH)
    plan = [(tool, json.loads(arguments)) for _task, _server, tool, arguments in scripted_wiring_check.scripted_steps(bench)]
    assert ("get_failure_modes", {"component": "bearing"}) in plan and ("submit_diagnosis", {"fan": 0, "belt": 100, "bearing": 2, "evidence_ids": ""}) in plan
    assert ("read_sensors", {"sensor_group": "basic", "cycles": 1}) in plan and ("read_sensors", {"sensor_group": "full", "cycles": 1}) in plan
    aob = [(tool, json.loads(arguments)) for _task, _server, tool, arguments in scripted_wiring_check.scripted_steps(bench, aob=True)]
    assert ("get_failure_modes", {"asset_class": "BEARING"}) in aob
    assert ("history", {"site_name": "Lab_3", "asset_id": BENCH, "sensors": ["T1_Temperature_C_1Hz"], "limit": 5}) in aob
    assert ("generate_work_order", {"description": "replace the bearing", "asset_num": "bench_1_bearing", "site_id": "Lab_3", "priority": 2,
                                    "work_type": "CM"}) in aob
    texts = json.dumps([scripted_wiring_check.scripted_steps(bench), scripted_wiring_check.scripted_steps(bench, aob=True)])
    assert not [name for name in [*rig_names(), *rig.default_config().sensor_groups] if re.search(rf"\b{re.escape(name)}\b", texts)]


def test_the_parts_bound_to_one_asset_refuse_another_asset(mounted, tmp_path):
    """The parts of the drivers that their modules bind to one asset, or to the records of one evidence kind, stop at launch by name
    on another, before any run folder exists: the greedy reference policy and the verifier scope all, which run the reference
    classifiers of the rig, the anchored scenarios, which lie on the dates of the rig's recordings, and the static agent on an
    asset whose records hold no recorded fields."""
    out = tmp_path / "runs"
    anchored = tmp_path / "anchored_bench.json"
    preset = json.loads((PACKAGE / "frameworks" / "AssetOpsBench" / "scenarios" / "anchored.json").read_text(encoding="utf-8"))
    anchored.write_text(json.dumps({**preset, "fault_rule": {"min_faulty": 1, "max_faulty": len(get_profile(BENCH).components)}}), encoding="utf-8")
    for asset_id, args, reason in (
            (BENCH, ["--config", "base", "--agent", "greedy"], "--agent greedy runs the reference classifiers"),
            (BENCH, ["--config", "base", "--agent", "scripted", "--verifier-scope", "all"], "the verifier scope all runs the reference classifiers"),
            (BENCH, ["--config", str(anchored), "--agent", "scripted"], "the anchored preset maps the scenarios of a scenario set"),
            (DEMO, ["--config", "base", "--agent", "static"], "the static agent shows the recorded fields")):
        with pytest.raises(SystemExit, match=f"{re.escape(reason)}.*(plays the asset {asset_id}|the asset {asset_id} that this run plays)"):
            run_episodes.main([*args, "--asset", asset_id, "--seeds", "1", "--store", "memory", "--out", str(out)])
    assert not out.exists()


def test_the_data_record_and_the_platform_record_name_the_asset(mounted, tmp_path):
    """The data record of a run names the data files of its asset: on the rig the files of before, on a port its data file, its own
    models file and the tables its package declares, and nothing of the rig's tables. The platform record names its asset. The
    keys that the record of every asset shares, those of the
    rig's record beyond its data file, its models and its two tables, come from the rig's record, so that the test holds whichever
    shared files a run records."""
    rig_facts = run_configuration.records.default_facts_path(get_profile(DEFAULT_ASSET))
    rig = ledger.data_files(rig_facts)
    own = ["facts", "models", "grades", "labels"]
    shared = [name for name in rig if name not in own]
    assert list(rig) == [*own, *shared]
    assert [rig[name]["path"] for name in ("models", "grades", "labels")] == [str(fit.MODELS_PATH), str(grade.GRADES_PATH), str(separability.LABELS_PATH)]
    bench_facts = run_configuration.records.default_facts_path(get_profile(BENCH))
    bench = ledger.data_files(bench_facts, BENCH)
    assert list(bench) == ["facts", "models", *shared] and bench["models"]["path"] == str(fit.models_path(BENCH))
    assert list(ledger.data_files(run_configuration.records.default_facts_path(get_profile(DEMO)), DEMO)) == ["facts", "models", "table", *shared]
    record = ledger.platform_command(asset_id=BENCH)
    assert record["asset_id"] == BENCH and record["data"]["facts"] == ledger.sha256_of(bench_facts)
    assert set(record["data"]) == {"facts", "models", *shared}


def test_a_root_with_runs_of_two_assets_stops_the_tables(tmp_path, monkeypatch):
    """A table set is the table set of one asset, since no cell of a table holds the asset: a root whose runs play two assets stops
    the table tools by name, and a root of one asset loads."""
    def run(name, asset_id):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "run.json").write_text(json.dumps({"asset_id": asset_id, "config": {"asset_id": asset_id, "max_steps": 30}}), encoding="utf-8")
        csv_rows.write_rows(folder / "scores.csv", [{"run": name, "seed": "0"}], ["run", "seed"])
        return folder

    runs = [run("rig_run", DEFAULT_ASSET), run("bench_run", BENCH)]
    monkeypatch.setattr(aggregate.pooling_rules, "folders_read", lambda root: runs)
    with pytest.raises(SystemExit, match=f"play the assets {BENCH}, {DEFAULT_ASSET}, and a table set is the table set of one asset"):
        aggregate.load_episodes(tmp_path)
    monkeypatch.setattr(aggregate.pooling_rules, "folders_read", lambda root: runs[1:])
    assert [episode["asset_id"] for episode in aggregate.load_episodes(tmp_path)] == [BENCH]


def test_precompute_and_the_reference_policies_follow_the_asset(mounted, tmp_path, capsys):
    """precompute_grid_tables.py computes the tables of one asset per call: on a port the models where the model code can fit it and no
    table of the rig, and a grid line of another asset stops it by name. run_reference_policies.py plays the policies that serve every asset
    on a port, every episode verified, and leaves out the greedy reference, whose classifiers are fitted on the rig."""
    rig_tables = [Path(separability.LABELS_PATH), Path(grade.GRADES_PATH)]
    before = [(path.exists(), path.stat().st_mtime_ns if path.exists() else None) for path in rig_tables]
    for asset_id in (DEMO, BENCH):
        assert precompute_tables.main(["--config", "base", "--asset", asset_id]) == {"labels": [], "grades": []}
    printed = capsys.readouterr().out
    assert f"no label table and no grade table for {BENCH}, whose label is not the label of diagnosability_label.py; models of the diagnostics server: fitted" in printed
    assert f"no label table and no grade table for {DEMO}" in printed and "models of the diagnostics server: none, since its evidence kind is demo_history" in printed
    assert [(path.exists(), path.stat().st_mtime_ns if path.exists() else None) for path in rig_tables] == before
    grid = tmp_path / "grid.txt"
    grid.write_text(f"--config base --asset {DEFAULT_ASSET} --agent scripted\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=f"plays the asset {DEFAULT_ASSET}, and this call computes the tables of the asset {BENCH}"):
        precompute_tables.main(["--grid", str(grid), "--asset", BENCH])
    out = tmp_path / "policies.csv"
    run_policies.main(["--config", "base", "--asset", BENCH, "--seeds", "2", "--out", str(out)])
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert sorted({row["policy"] for row in rows}) == ["declare", "do_nothing", "oracle", "random"] and len(rows) == 8
    assert {row["verified"] for row in rows} == {"True"}


def _configurations_of(tmp_path, line, asset_id):
    """The configurations that the tables call of asset_id resolves for a grid file of the one line."""
    grid = tmp_path / "grid.txt"
    grid.write_text(line + "\n", encoding="utf-8")
    _facts, _cycles, blocks = run_configuration.load_world(None, asset_id)
    return precompute_tables.configurations([str(grid)], [], blocks=blocks, asset_id=asset_id)


def test_a_grid_line_that_names_no_asset_takes_the_asset_of_the_tables_call(mounted, tmp_path):
    """The tables call adds its --asset to a line that names none, so the line has the tables of the asset of the call; the
    third case, a line of another asset, stops the call by name in test_precompute_and_the_reference_policies_follow_the_asset."""
    [(_name, config)] = _configurations_of(tmp_path, "--config base --agent scripted", BENCH)
    assert config.asset_id == BENCH


def test_a_grid_line_keeps_the_asset_that_it_names(mounted, tmp_path):
    """A line that names the asset of the call keeps its own words and has the tables of that asset, in either form of the
    option."""
    for line in (f"--config base --asset {BENCH} --agent scripted", f"--config base --asset={BENCH} --agent scripted"):
        [(name, config)] = _configurations_of(tmp_path, line, BENCH)
        assert config.asset_id == BENCH and name.endswith(f": {line}"), line


def test_the_judge_reads_the_asset_of_its_episode(mounted, tmp_path):
    """The reference answer of the judge is composed from the asset that released.json of the episode names: its components, the
    codes of their levels, its work orders and its noun, and no name of hyd_1."""
    ep = episode_folder(tmp_path)
    released = read_json(ep / "released.json")
    (ep / "released.json").write_text(json.dumps({**released, "asset_id": BENCH}), encoding="utf-8")
    (ep / "hidden" / "oracle.json").write_text(json.dumps({"diagnosis": {"fan": 1, "belt": 100, "bearing": 1},
                                                           "work_orders": [{"component": "fan", "window_end": 9}], "finish": True}), encoding="utf-8")
    reference = judge.judge_episode(ep, FakeJudge([RUBRIC_PASS]), "single")["reference_answer"]
    assert "the fan at level 1 (faulty, code fan_stalled)" in reference and "The faulty components are fan." in reference
    assert "one work order on the fan closed by cycle 9" in reference and "The bench is restored" in reference
    assert not [name for name in [*rig_names(), "rig"] if re.search(rf"\b{re.escape(name)}\b", reference)]


def test_the_runs_of_a_port_lie_in_the_folder_of_its_port():
    """Without --out the runs of the engine's default asset go to the runs folder of the rig and those of any other asset to the
    run folders of its port, which the code lists leave out, so that no root holds the runs of two assets by default."""
    assert run_episodes.build_parser().parse_args([]).out is None
    assert run_episodes.default_out(DEFAULT_ASSET) == "CausalDiagnosis/results/runs"
    assert run_episodes.default_out(BENCH) == "CausalDiagnosis/results/runs/bench_1"
