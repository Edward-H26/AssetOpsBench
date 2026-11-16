"""The session module on real server processes over stdio, without a model. Every server is a stand-in that the test writes
into its temporary folder, with tools that count, fail, answer late, block, end their process or break the protocol stream."""
import asyncio
import json
import os
import signal
import time
from pathlib import Path

import pytest

from CausalDiagnosis.utils import model_request_retry as retry
from CausalDiagnosis.agents.utils.mcp_sessions import TEARDOWN, ToolTransportError, open_sessions

STAND_IN = '''
import asyncio
import os
import signal
import sys
import time
from pathlib import Path

Path(FOLDER, NAME + ".pid").write_text(str(os.getpid()), encoding="utf-8")
time.sleep(START_DELAY_S)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stand-in-" + NAME, log_level="ERROR")
COUNT = [0]


@mcp.tool()
def count() -> dict:
    """Count the calls this process has answered."""
    COUNT[0] += 1
    return {"server": NAME, "pid": os.getpid(), "count": COUNT[0]}


@mcp.tool()
def fail() -> dict:
    """Raise."""
    raise ValueError("the tool failed on purpose")


@mcp.tool()
def typed(n: int) -> dict:
    """Take a whole number."""
    return {"n": n}


@mcp.tool()
async def slow(seconds: float, path: str) -> dict:
    """Answer late, and write a file when the answer is due."""
    await asyncio.sleep(seconds)
    Path(path).write_text("late", encoding="utf-8")
    return {"slept": seconds}


@mcp.tool()
def blocking(seconds: float, path: str) -> dict:
    """Hold the whole server for a while, then write a file and answer."""
    time.sleep(seconds)
    Path(path).write_text("late", encoding="utf-8")
    return {"slept": seconds}


@mcp.tool()
def die() -> dict:
    """End the process without an answer."""
    os._exit(3)


@mcp.tool()
def crash() -> dict:
    """End the process by a segmentation fault, as a crash does."""
    os.kill(os.getpid(), signal.SIGSEGV)
    return {}


@mcp.tool()
def late_exit(seconds: float) -> dict:
    """Hold the whole server for a while, then end with a traceback and a nonzero code, as a write to a closed pipe can."""
    time.sleep(seconds)
    sys.stderr.write("Traceback (most recent call last):\\n  BrokenPipeError: the answer could not be written\\n")
    sys.stderr.flush()
    os._exit(1)


@mcp.tool()
async def garble() -> dict:
    """Write bytes that are no text onto the protocol stream, then answer late."""
    sys.stdout.buffer.write(bytes([255, 254, 250, 10]))
    sys.stdout.buffer.flush()
    await asyncio.sleep(30)
    return {}


mcp.run(transport="stdio")
'''


def stand_in(folder, name, start_delay_s=0.0):
    """The launcher of one stand-in server, which writes its process id into folder/<name>.pid as it starts."""
    path = Path(folder) / f"{name}_server.py"
    path.write_text(f"NAME, FOLDER, START_DELAY_S = {name!r}, {str(folder)!r}, {start_delay_s!r}\n{STAND_IN}", encoding="utf-8")
    return path


def pid_of(folder, name):
    return int((Path(folder) / f"{name}.pid").read_text(encoding="utf-8"))


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def answer(pair):
    text, is_error = pair
    assert is_error is False, text
    return json.loads(text)


def test_one_process_serves_every_call_of_a_server_and_each_server_has_its_own(tmp_path):
    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a"), "b": stand_in(tmp_path, "b")}) as sessions:
            return [answer(await sessions.call(server, "count", {})) for server in ("a", "a", "b")]

    first, second, other = asyncio.run(play())
    assert first["pid"] == second["pid"] == pid_of(tmp_path, "a") and (first["count"], second["count"]) == (1, 2)
    assert other["pid"] == pid_of(tmp_path, "b") != first["pid"] and other["count"] == 1
    assert gone(first["pid"]) and gone(other["pid"])  # the block ended both processes


def test_the_tools_of_every_server_are_listed_when_the_sessions_open(tmp_path):
    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a"), "b": stand_in(tmp_path, "b")}) as sessions:
            return sessions.tools

    tools = asyncio.run(play())
    assert list(tools) == ["a", "b"]  # in the order the servers were given
    for listed in tools.values():
        assert sorted(tool.name for tool in listed) == ["blocking", "count", "crash", "die", "fail", "garble", "late_exit", "slow", "typed"]
    typed = next(tool for tool in tools["a"] if tool.name == "typed")
    assert typed.description == "Take a whole number." and typed.inputSchema["properties"]["n"]["type"] == "integer"


def test_an_error_answer_comes_back_as_text_with_its_mark_and_raises_nothing(tmp_path):
    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a")}) as sessions:
            return [await sessions.call("a", tool, arguments) for tool, arguments in (("fail", {}), ("not_listed", {}), ("typed", {"n": "x"}))]

    raised, unlisted, refused = asyncio.run(play())
    assert raised[1] is True and "the tool failed on purpose" in raised[0]
    assert unlisted[1] is True and "not_listed" in unlisted[0]
    assert refused[1] is True and "valid integer" in refused[0]


def test_a_call_past_the_deadline_raises_a_transport_error_and_its_server_is_gone_when_the_block_ends(tmp_path):
    late = tmp_path / "late.txt"

    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a")}) as sessions:
            started = time.perf_counter()
            with pytest.raises(ToolTransportError, match=r"^slow of a gave no answer before the deadline of the episode$"):
                with retry.episode_deadline(time.monotonic() + 0.3):
                    await sessions.call("a", "slow", {"seconds": 30.0, "path": str(late)})
            waited = time.perf_counter() - started
            return waited, answer(await sessions.call("a", "count", {}))  # the session still answers the next call

    waited, after = asyncio.run(play())
    assert waited < 5 and after["count"] == 1
    assert gone(after["pid"]) and not late.exists()  # no late action can land once the block has ended


def test_a_server_that_died_raises_a_transport_error_on_that_call_and_on_the_next(tmp_path):
    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a")}) as sessions:
            started = time.perf_counter()
            errors = []
            for tool in ("die", "count"):
                with pytest.raises(ToolTransportError) as caught:
                    await sessions.call("a", tool, {})
                errors.append(str(caught.value))
            return errors, time.perf_counter() - started

    errors, waited = asyncio.run(play())
    assert errors[0].startswith("die of a got no answer: ") and errors[1].startswith("count of a got no answer: ")
    assert waited < 10  # long before the limit of a call


def test_a_call_on_closed_sessions_or_to_a_server_the_block_did_not_open_raises_a_transport_error(tmp_path):
    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a")}) as sessions:
            with pytest.raises(ToolTransportError, match="^count of b got no answer: the block opened no server of that name$"):
                await sessions.call("b", "count", {})
        with pytest.raises(ToolTransportError, match="^count of a got no answer: the sessions are closed$"):
            await sessions.call("a", "count", {})

    asyncio.run(play())


def test_a_server_that_does_not_start_raises_a_transport_error_that_names_it(tmp_path):
    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a"), "ghost": tmp_path / "no_such_server.py"}):
            raise AssertionError("the block opened")

    with pytest.raises(ToolTransportError, match="^the server ghost did not start: ") as caught:
        asyncio.run(play())
    assert caught.value.__cause__ is not None  # what the client raised stays behind the error
    assert gone(pid_of(tmp_path, "a"))  # the server that had started is ended too


def test_an_error_of_the_callers_block_leaves_the_block_as_itself(tmp_path):
    cause = KeyError("the cause")

    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a")}):
            raise ValueError("the caller's own error") from cause

    with pytest.raises(ValueError, match="^the caller's own error$") as caught:
        asyncio.run(play())
    assert caught.value.__cause__ is cause


def test_a_cut_block_ends_as_a_time_out_of_its_caller(tmp_path):
    async def call_pending():
        async with open_sessions({"a": stand_in(tmp_path, "a")}) as sessions:
            await sessions.call("a", "slow", {"seconds": 30.0, "path": str(tmp_path / "late.txt")})

    async def opening_pending():
        async with open_sessions({"b": stand_in(tmp_path, "b", start_delay_s=30.0)}):
            raise AssertionError("the block opened")

    async def play(block, limit_s):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(block(), limit_s)
        return asyncio.current_task().cancelling()

    assert asyncio.run(play(call_pending, 3.0)) == 0 and gone(pid_of(tmp_path, "a"))
    assert asyncio.run(play(opening_pending, 1.0)) == 0 and gone(pid_of(tmp_path, "b"))
    assert not (tmp_path / "late.txt").exists()


def test_a_signal_that_the_teardown_sent_is_marked_and_a_signal_before_the_end_is_not(tmp_path):
    """A cut while the server is busy in a tool longer than the client's wait after stdin closes: the client ends the process by
    a signal, and the block marks that return code as its own teardown's. The control: a server killed by a signal while the
    block runs keeps an unmarked code, which the evidence still reads as a fault of the server."""
    busy, killed = tmp_path / "busy", tmp_path / "killed"
    busy.mkdir()
    killed.mkdir()

    async def cut_while_busy(cut):
        async with open_sessions({"a": stand_in(busy, "a")}) as sessions:
            cut.reschedule(asyncio.get_running_loop().time() + 1.0)  # counted from the open of the block, not from the start of the server
            await sessions.call("a", "blocking", {"seconds": 8.0, "path": str(busy / "late.txt")})

    async def killed_in_the_block():
        async with open_sessions({"a": stand_in(killed, "a")}):
            os.kill(pid_of(killed, "a"), signal.SIGKILL)
            await asyncio.sleep(0.5)

    async def play():
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(None) as cut:
                await cut_while_busy(cut)
        await killed_in_the_block()

    asyncio.run(play())
    busy_lines = (busy / "hidden" / "a.0.returncode").read_text(encoding="utf-8").split()
    killed_lines = (killed / "hidden" / "a.0.returncode").read_text(encoding="utf-8").split()
    assert int(busy_lines[0]) < 0 and busy_lines[1:] == [TEARDOWN]
    assert int(killed_lines[0]) < 0 and killed_lines[1:] == []


def test_a_crash_during_a_call_is_no_teardown_and_an_exit_after_the_close_is(tmp_path):
    """A server that crashes during a call, with no cut, unwinds the block before asyncio may have set its return code; its
    end came before the close, so its code stays unmarked and the evidence reads it as a fault. A server still running at the
    close that ends with a traceback and a nonzero code 0.5 s after the cut, inside the client's wait, is marked, so the cut stays
    the agent's. The control of both is the earlier test: a signal inside the block stays unmarked."""
    crashed, late = tmp_path / "crashed", tmp_path / "late"
    crashed.mkdir()
    late.mkdir()

    async def crash_in_a_call():
        async with open_sessions({"a": stand_in(crashed, "a")}) as sessions:
            await sessions.call("a", "crash", {})

    async def cut_before_the_exit(cut):
        async with open_sessions({"a": stand_in(late, "a")}) as sessions:
            cut.reschedule(asyncio.get_running_loop().time() + 1.0)  # counted from the open of the block, not from the start of the server
            await sessions.call("a", "late_exit", {"seconds": 1.5})

    async def play():
        with pytest.raises(ToolTransportError):
            await crash_in_a_call()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(None) as cut:
                await cut_before_the_exit(cut)

    asyncio.run(play())
    crashed_lines = (crashed / "hidden" / "a.0.returncode").read_text(encoding="utf-8").split()
    late_lines = (late / "hidden" / "a.0.returncode").read_text(encoding="utf-8").split()
    assert crashed_lines == [str(-signal.SIGSEGV)]
    assert late_lines == ["1", TEARDOWN]


def test_the_close_and_the_mark_hold_without_os_waitid_and_a_failed_query_still_closes(tmp_path, monkeypatch):
    """An os without waitid (macOS before Python 3.13) still closes the sessions and marks a server cut while busy, since the
    kernel is asked through the C library there; a query of the kernel that fails, by an OSError or by any other error, still
    closes the sessions and writes the code, unmarked. The control is the test of the mark above, on the os of the machine."""
    from types import SimpleNamespace

    from CausalDiagnosis.agents.utils import mcp_sessions

    plain = SimpleNamespace(**{name: getattr(os, name) for name in dir(os) if name != "waitid" and not name.startswith("__")})
    if mcp_sessions.sys.platform != "darwin":
        pytest.skip("the C library's waitid is read for the layout of macOS alone; Linux has os.waitid")

    def failing(error):
        def query(*_args):
            raise error
        return query

    async def cut_while_busy(folder, opened, cut):
        async with open_sessions({"a": stand_in(folder, "a")}) as sessions:
            opened.append(sessions)
            cut.reschedule(asyncio.get_running_loop().time() + 1.0)  # counted from the open of the block, not from the start of the server
            await sessions.call("a", "blocking", {"seconds": 8.0, "path": str(folder / "late.txt")})

    async def play(folder):
        opened = []
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(None) as cut:
                await cut_while_busy(folder, opened, cut)
        return opened[0]

    without = tmp_path / "without"
    without.mkdir()
    monkeypatch.setattr(mcp_sessions, "os", plain)
    sessions = asyncio.run(play(without))
    lines = (without / "hidden" / "a.0.returncode").read_text(encoding="utf-8").split()
    assert sessions._open is False and int(lines[0]) < 0 and lines[1:] == [TEARDOWN]
    # an OSError reads as ended in _running; any other error, as the missing os.waitid raised, reaches the guard of _running_ids
    for error in (OSError("the query failed"), AttributeError("the query is missing")):
        folder = tmp_path / type(error).__name__
        folder.mkdir()
        monkeypatch.setattr(mcp_sessions, "_waitid", failing(error))
        sessions = asyncio.run(play(folder))
        lines = (folder / "hidden" / "a.0.returncode").read_text(encoding="utf-8").split()
        assert sessions._open is False and int(lines[0]) < 0 and lines[1:] == [], error


def test_a_call_from_a_worker_thread_is_answered_on_the_loop(tmp_path):
    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a")}) as sessions:
            loop = asyncio.get_running_loop()
            return await asyncio.to_thread(lambda: asyncio.run_coroutine_threadsafe(sessions.call("a", "count", {}), loop).result())

    assert answer(asyncio.run(play()))["count"] == 1


def test_a_call_that_another_task_left_in_flight_raises_a_transport_error_when_the_block_ends(tmp_path):
    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a")}) as sessions:
            left = asyncio.ensure_future(sessions.call("a", "slow", {"seconds": 30.0, "path": str(tmp_path / "late.txt")}))
            await asyncio.sleep(0.5)
        # read when the block has ended, before asyncio.run cancels what is left, which would end the call too
        return left, left.done()

    started = time.perf_counter()
    left, done_when_the_block_ended = asyncio.run(play())
    assert time.perf_counter() - started < 20  # given up with the block, not at the limit of the call
    assert done_when_the_block_ended and not left.cancelled()  # it had raised before the block ended
    with pytest.raises(ToolTransportError, match="^slow of a got no answer: the sessions closed before it came$"):
        left.result()


def test_a_server_that_still_writes_while_it_is_ended_does_not_make_a_finished_block_raise(tmp_path):
    late = tmp_path / "late.txt"

    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a")}) as sessions:
            with pytest.raises(ToolTransportError, match="gave no answer before the deadline"):
                with retry.episode_deadline(time.monotonic() + 0.3):
                    await sessions.call("a", "blocking", {"seconds": 1.0, "path": str(late)})
        return "finished"

    assert asyncio.run(play()) == "finished"
    assert late.exists()  # the server did answer while it was being ended, which is what used to raise
    assert gone(pid_of(tmp_path, "a"))


def test_a_transport_that_breaks_under_the_caller_raises_a_transport_error_and_no_cancellation(tmp_path):
    async def play():
        try:
            async with open_sessions({"a": stand_in(tmp_path, "a")}) as sessions:
                await sessions.call("a", "garble", {})
        finally:
            assert asyncio.current_task().cancelling() == 0

    started = time.perf_counter()
    with pytest.raises(ToolTransportError, match="^the sessions broke under the caller: UnicodeDecodeError") as caught:
        asyncio.run(play())
    assert time.perf_counter() - started < 20 and isinstance(caught.value.__cause__, BaseExceptionGroup)
    assert gone(pid_of(tmp_path, "a"))
