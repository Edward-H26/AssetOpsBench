"""The transcript of an episode (servers/transcript.py, CausalDiagnosis/mcp_servers/utils/transcript_hooks.py): the numbering of its lines under one lock,
the reader that refuses a torn or edited file, the wrapper at the two handlers of the MCP package, and the lines of the action
server, each with the hash of the text its client received. No test states a value computed from the rig data."""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

mcp_types = pytest.importorskip("mcp.types")  # the MCP package is pinned in the fork's uv.lock

from mcp.server.fastmcp import FastMCP  # noqa: E402

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig  # noqa: E402
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import HEALTHY, default_facts_path  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.mcp_servers.utils import transcript  # noqa: E402
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode  # noqa: E402
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import build_server  # noqa: E402
from CausalDiagnosis.utils.atomic_writes_and_locks import locked  # noqa: E402
from CausalDiagnosis.mcp_servers.utils.transcript_hooks import lowlevel_of, sent_text, transcribed  # noqa: E402

from CausalDiagnosis.utils.paths import ROOT, SRC
needs_facts = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")
# a reading, an order before any diagnosis (refused, and a step), a diagnosis, the finish, a call after the end (no step)
SEQUENCE = [
    ("read_sensors", {"sensor_group": "tier1", "cycles": 1}),
    ("create_work_order", {"component": "cooler", "prediction_id": "", "evidence_ids": ""}),
    ("submit_diagnosis", {**HEALTHY, "evidence_ids": ""}),
    ("finish", {"summary": "done"}),
    ("get_current_time", {}),
]
START_WALL = 100.0


def ask(app, tool, arguments):
    """One tool call through the handler that the MCP package calls for a client. Returns the result the client would get."""
    request = mcp_types.CallToolRequest(method="tools/call", params=mcp_types.CallToolRequestParams(name=tool, arguments=arguments))
    return asyncio.run(lowlevel_of(app).request_handlers[mcp_types.CallToolRequest](request)).root


def listing(app):
    """One tool listing through the handler that the MCP package calls for a client."""
    request = mcp_types.ListToolsRequest(method="tools/list")
    return asyncio.run(lowlevel_of(app).request_handlers[mcp_types.ListToolsRequest](request)).root.tools


def play(folder, calls=SEQUENCE):
    """An episode of the action server in the folder under a recorded clock, one second per call: (the state path, the
    result of every call, the hash of the released evidence at the start and after every call)."""
    path = folder / "episode.json"
    with episode.recorded_clock(START_WALL, 5.0):
        episode.start(path, 3, EpisodeConfig(), default_facts_path())
        episode.hand_over(path)
    app = build_server(path)
    answers, released = [], [transcript.file_sha256(episode.released_path(path))]
    for k, (tool, arguments) in enumerate(calls, start=1):
        with episode.recorded_clock(START_WALL + k, 5.0 + k):
            answers.append(ask(app, tool, arguments))
        released.append(transcript.file_sha256(episode.released_path(path)))
    return path, answers, released


def test_lines_are_numbered_in_append_order_and_a_torn_or_renumbered_file_is_refused(tmp_path):
    assert transcript.begin(tmp_path, wall=1.0) == 0
    assert transcript.append(tmp_path, "list", server="s", listing_sha256="x") == 1
    assert transcript.answered(tmp_path, "s", "t", {"a": 1}, "text", False) == 2
    lines = transcript.read(tmp_path)
    assert [(line["n"], line["kind"]) for line in lines] == [(0, "start"), (1, "list"), (2, "call")]
    assert lines[0]["format"] == transcript.FORMAT and all(line["pid"] == os.getpid() for line in lines)
    assert lines[2]["answer_sha256"] == transcript.text_sha256("text") and lines[2]["stepped"] is False and "answer" not in lines[2]
    path = tmp_path / transcript.NAME
    whole = path.read_text(encoding="utf-8")
    path.write_text(whole[:-5], encoding="utf-8")  # the append of the last line was cut off
    with pytest.raises(ValueError, match="line 2 is torn"):
        transcript.read(tmp_path)
    renumbered = [json.loads(text) for text in whole.splitlines()]
    renumbered[1]["n"] = 5
    path.write_text("".join(json.dumps(line) + "\n" for line in renumbered), encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 is torn"):
        transcript.read(tmp_path)
    # a new episode in the folder starts a new record
    assert transcript.begin(tmp_path, wall=2.0) == 0 and [line["kind"] for line in transcript.read(tmp_path)] == ["start"]


def test_an_answer_text_with_any_line_separator_stays_one_line(tmp_path):
    text = "first" + chr(0x2028) + "second\nthird" + chr(0x85) + "fourth"
    transcript.answered(tmp_path, "s", "t", {}, text, False, keep_text=True)
    [line] = transcript.read(tmp_path)
    assert line["answer"] == text and line["answer_sha256"] == transcript.text_sha256(text)
    assert (tmp_path / transcript.NAME).read_bytes().count(b"\n") == 1


def test_an_append_waits_for_the_lock_of_the_transcript_and_two_processes_keep_every_line(tmp_path):
    transcript.begin(tmp_path)
    # each writer says when it is about to append, so the wait below starts only once both stand at the lock
    script = ("import sys\nfrom pathlib import Path\nfrom CausalDiagnosis.mcp_servers.utils import transcript\n"
              "Path(sys.argv[1], 'ready_' + sys.argv[2]).write_text('')\n"
              "for k in range(50):\n    transcript.append(sys.argv[1], 'call', writer=sys.argv[2], k=k)\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(ROOT), str(SRC), os.environ.get("PYTHONPATH", "")) if p)}
    path = tmp_path / transcript.NAME
    with locked(path):  # while this process holds the lock of the transcript, no other process appends a line
        writers = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path), name], env=env) for name in ("a", "b")]
        deadline = time.monotonic() + 120
        while not all((tmp_path / f"ready_{name}").exists() for name in ("a", "b")):
            assert time.monotonic() < deadline, "the writers did not start"
            time.sleep(0.01)
        held_until = time.monotonic() + 1.0
        while time.monotonic() < held_until:
            assert path.read_bytes().count(b"\n") == 1, "a process appended a line while another held the lock"
            time.sleep(0.01)
    assert [writer.wait(timeout=120) for writer in writers] == [0, 0]
    lines = transcript.read(tmp_path)  # the reader itself checks that the numbers are 0, 1, 2 and so on
    assert len(lines) == 101 and len({line["pid"] for line in lines[1:]}) == 2
    for name in ("a", "b"):
        assert [line["k"] for line in lines if line.get("writer") == name] == list(range(50))


def test_the_wrapper_records_a_listing_and_every_call_through_the_handlers_of_the_mcp_package(tmp_path):
    app = FastMCP("probe")

    @app.tool()
    def echo(word: str) -> str:
        return word

    assert transcribed(app, tmp_path, "probe", keep_text=True) is app
    tools = listing(app)
    answered = ask(app, "echo", {"word": "hello"})
    failed = ask(app, "no_such_tool", {})
    lines = transcript.read(tmp_path)
    # the package lists the tools by itself before a call, to fill its cache, and that listing adds no line
    assert [line["kind"] for line in lines] == ["list", "call", "call"]
    assert [tool.name for tool in tools] == ["echo"] and lines[0] == {**lines[0], "server": "probe", "listing_sha256": transcript.listing_sha256(tools)}
    assert (lines[1]["tool"], lines[1]["arguments"], lines[1]["is_error"], lines[1]["answer"]) == ("echo", {"word": "hello"}, False, "hello")
    assert lines[1]["answer_sha256"] == transcript.text_sha256(sent_text(answered))
    assert failed.isError and (lines[2]["tool"], lines[2]["is_error"], lines[2]["stepped"]) == ("no_such_tool", True, False)
    assert lines[2]["answer"] == sent_text(failed) and lines[2]["answer_sha256"] == transcript.text_sha256(sent_text(failed))


def test_a_listing_hash_follows_the_text_of_a_tool(tmp_path):
    def served(description):
        app = FastMCP("probe")

        def echo(word: str) -> str:
            return word

        app.tool(description=description)(echo)
        return transcript.listing_sha256(listing(app))

    assert served("Repeat a word.") == served("Repeat a word.") != served("Repeat one word.")


def test_a_server_without_an_episode_folder_writes_nothing(tmp_path):
    app = FastMCP("probe")
    handlers = dict(lowlevel_of(app).request_handlers)
    assert transcribed(app, None, "probe") is app and lowlevel_of(app).request_handlers == handlers
    assert transcript.append(None, "list") is None and transcript.append(tmp_path / "missing", "list") is None
    assert transcript.begin(tmp_path / "missing") is None and transcript.read(tmp_path) == [] and list(tmp_path.iterdir()) == []
    transcript.note(released_sha256="x")  # outside a served call a note goes nowhere


@needs_facts
def test_every_call_of_the_action_server_enters_the_transcript_with_the_hash_of_its_answer(tmp_path):
    calls = [*SEQUENCE, ("no_such_tool", {})]
    path, answers, released = play(tmp_path, calls)
    start, handover, *lines = transcript.read(tmp_path)
    assert (start["kind"], start["wall"], start["released_sha256"]) == ("start", START_WALL, released[0])
    assert (handover["kind"], handover["wall"], handover["mono"]) == ("handover", START_WALL, 5.0)
    # every call that reaches the action server is entered and exited around its line, and the call of no tool never reaches it
    assert [line["kind"] for line in lines] == ["enter", "call", "exit"] * len(SEQUENCE) + ["call"]
    lines = [line for line in lines if line["kind"] == "call"]
    assert [(line["kind"], line["server"], line["tool"], line["arguments"]) for line in lines] == [
        ("call", contract.ACTION_SERVER, tool, arguments) for tool, arguments in calls]
    # the hash of every line is the hash of the text the client received, and of the text the state file keeps
    assert [line["answer_sha256"] for line in lines] == [transcript.text_sha256(sent_text(answer)) for answer in answers]
    stored = episode.load_state(path)["tool_calls"]
    assert [transcript.text_sha256(entry["response"]) for entry in stored] == [line["answer_sha256"] for line in lines[:5]]
    assert "refused" in json.loads(sent_text(answers[1])) and "refused" in json.loads(sent_text(answers[4]))
    # the refused order took a step, the call after the end and the call of no tool took none
    assert [line["stepped"] for line in lines] == [True, True, True, True, False, False]
    assert [line["is_error"] for line in lines] == [False, False, False, False, False, True] and answers[5].isError
    assert [line["wall"] for line in lines[:5]] == [START_WALL + k for k in range(1, 6)] and "wall" not in lines[5]
    assert [line["released_sha256"] for line in lines[:5]] == released[1:6]
    assert not any("answer" in line for line in lines)  # the action server keeps its answers in the state file


@needs_facts
def test_the_state_file_and_the_released_evidence_keep_their_bytes_with_the_transcript(tmp_path, monkeypatch):
    def files(folder):
        folder.mkdir()
        path, _answers, _released = play(folder)
        return path.read_bytes(), episode.released_path(path).read_bytes()

    recorded = files(tmp_path / "recorded")
    monkeypatch.setattr(transcript, "begin", lambda *args, **fields: None)
    monkeypatch.setattr(transcript, "append", lambda *args, **fields: None)
    bare = files(tmp_path / "bare")
    assert bare[1] == recorded[1]
    # the thinking clock of the state reads the transcript: the byte offset read so far and the line of the call in service,
    # which name nothing without a transcript, and every other field of the state keeps its value
    state, bare_state = json.loads(recorded[0]), json.loads(bare[0])
    assert (bare_state.pop("transcript_cursor"), bare_state["open_service"].pop("n")) == (0, None)
    assert state.pop("transcript_cursor") > 0 and state["open_service"].pop("n") is not None and bare_state == state
    assert (tmp_path / "recorded" / transcript.NAME).exists() and not (tmp_path / "bare" / transcript.NAME).exists()


def test_an_atomic_write_leaves_no_temporary_file_when_it_fails_and_gives_the_mode_of_the_umask(tmp_path):
    """write_json_atomically replaces the file through a temporary file beside it: a dump that raises leaves the folder as it
    was, the old file included, and a written file takes the mode an ordinary new file takes under the umask of the process."""
    from CausalDiagnosis.utils.atomic_writes_and_locks import write_json_atomically

    path = tmp_path / "state.json"
    write_json_atomically(path, {"n": 1})
    with pytest.raises(TypeError):
        write_json_atomically(path, {"n": object()})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"] and json.loads(path.read_text(encoding="utf-8")) == {"n": 1}
    umask = os.umask(0)
    os.umask(umask)
    assert path.stat().st_mode & 0o777 == 0o666 & ~umask
