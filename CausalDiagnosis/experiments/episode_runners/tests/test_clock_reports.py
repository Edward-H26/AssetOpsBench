"""The launcher's stamp and isolation, the server baseline, the end of a cut at the wall
limit, and the clock cells of a row with their sums in run.json."""
import json
import subprocess
import sys

from CausalDiagnosis.experiments.episode_runners import episode_launchers, servers_of_a_run
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
from CausalDiagnosis.experiments.episode_runners import run_episodes


def test_the_launcher_stamps_its_start_first_and_refuses_a_process_that_is_not_isolated(tmp_path):
    launchers = episode_launchers.write_launchers(tmp_path, servers=(servers_of_a_run.RIG_SERVER,))
    text = launchers[servers_of_a_run.RIG_SERVER].read_text(encoding="utf-8")
    lines = text.splitlines()
    first = next(i for i, line in enumerate(lines) if line.rstrip().endswith('"""')) + 1  # the first statement after the docstring
    assert lines[first:first + 2] == ["import time", "_STARTED = (time.time(), time.monotonic(), time.thread_time())  # the start s of the process for the thinking clock"]
    assert text.rstrip().endswith("started=_STARTED)")
    plain = subprocess.run([sys.executable, str(launchers[servers_of_a_run.RIG_SERVER])], capture_output=True, text=True, timeout=60)
    assert plain.returncode != 0 and "runs only isolated" in plain.stderr


def test_the_baseline_names_each_launcher_hash_and_the_measured_threads(tmp_path):
    launchers = episode_launchers.write_launchers(tmp_path, servers=(servers_of_a_run.RIG_SERVER, servers_of_a_run.CAUSAL_SERVER))
    run_episodes.write_server_baseline(tmp_path, launchers, {servers_of_a_run.RIG_SERVER: 3})
    baseline = json.loads((tmp_path / "hidden" / "server_baseline.json").read_text(encoding="utf-8"))
    from CausalDiagnosis.mcp_servers.utils import transcript
    assert baseline[servers_of_a_run.RIG_SERVER] == {"launcher_sha256": transcript.file_sha256(launchers[servers_of_a_run.RIG_SERVER]), "threads": 3}
    assert set(baseline[servers_of_a_run.CAUSAL_SERVER]) == {"launcher_sha256"}


def test_the_end_of_a_cut_decides_its_class():
    assert failures.attempt_class([], cut=True, cut_end=failures.SERVICE)[0] == failures.INFRASTRUCTURE
    assert failures.attempt_class([], cut=True, cut_end=failures.WALL)[0] == failures.AGENT


def test_an_open_tool_call_is_an_entry_without_its_exit(tmp_path):
    from CausalDiagnosis.mcp_servers.utils import transcript
    first = transcript.append(tmp_path, "enter", server="rig", pid=1, tool="read_sensors", wall=0, mono=0, thread_cpu=0)
    second = transcript.append(tmp_path, "enter", server="rig", pid=1, tool="read_sensors", wall=0, mono=0, thread_cpu=0)
    transcript.append(tmp_path, "exit", server="rig", pid=1, enter_n=first, outcome="answered")
    assert run_episodes.open_tool_calls(tmp_path) == {second}


def test_a_no_call_step_open_at_the_cut_is_no_open_tool_call(tmp_path):
    """The driver's own no_call writes an entry line like a counted call; a cut inside it leaves no tool call of the agent open,
    so with no failed span and no model request open the cut ends as wall, not as service."""
    from CausalDiagnosis.mcp_servers.utils import transcript
    from CausalDiagnosis.utils.public_rules.visits import NO_CALL_ACTION
    from CausalDiagnosis.experiments.episode_runners import wall_cut
    transcript.append(tmp_path, "enter", server="rig", pid=1, tool=NO_CALL_ACTION, wall=0, mono=0, thread_cpu=0)
    assert run_episodes.open_tool_calls(tmp_path) == set()
    assert wall_cut.cut_end(False, bool(run_episodes.open_tool_calls(tmp_path)), False)["end"] == failures.WALL
    opened = transcript.append(tmp_path, "enter", server="rig", pid=1, tool="read_sensors", wall=0, mono=0, thread_cpu=0)
    assert run_episodes.open_tool_calls(tmp_path) == {opened}


def test_the_run_sums_the_clock_cells_of_its_rows():
    rows = [{"credited_s": '{"starts":1.5,"calls":2.0}', "uncredited_s": '{"starts":0.5,"calls":1.0}', "failed_model_s": 3.0,
             "failed_model_attempts": 2, "uncharged_after_last_s": 0.25, "cut_end": ""},
            {"credited_s": '{"starts":1.0,"calls":0.5}', "uncredited_s": '{"starts":0.0,"calls":0.5}', "failed_model_s": "",
             "failed_model_attempts": "", "uncharged_after_last_s": 1.0, "cut_end": "service"}]
    totals = run_episodes.clock_totals(rows)
    assert totals["model_failures"] == {"count": 2, "seconds": 3.0}
    assert totals["credited_seconds"] == {"starts": 2.5, "calls": 2.5}
    assert totals["uncredited_seconds"] == {"starts": 0.5, "calls": 1.5}
    assert totals["uncharged_after_last_seconds"] == 1.25 and totals["cuts"] == {"service": 1, "wall": 0}
