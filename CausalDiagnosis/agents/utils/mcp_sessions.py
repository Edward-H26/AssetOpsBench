"""One MCP session per server for the length of a block, for every runner and for the drivers.

A block of open_sessions starts each server of a run once, as ``python -I <launcher path>`` (isolated: no environment variable
and no user site of Python reaches the server), lists its tools, and ends the processes when it ends. Each process writes its error
output to hidden/<server>.<k>.stderr of the episode folder that holds its launcher, k the start index of that server there, and the
block records the return code of every process it started (EpisodeSessions.returncodes, and hidden/<server>.<k>.returncode beside
its log), which attempt_error_classes.py reads. Every agent and every call of the harness play on such a block, so no agent pays for the start of a
server that another does not pay for, and one rule says what a tool call without an answer is.

A call has one time limit, which covers the sending too. A call past it, a call on closed sessions, a call that is still
in flight when the block ends, a call to a server whose process is gone or that the block did not open, and a reply of the
protocol that is no tool answer raise ToolTransportError. An answer that the server marks as an error (a refusal, an
exception of the tool, an argument its schema refuses, a tool it does not list) is an answer and comes back as its text with
the mark. The client gives a call up and cannot recall it, so the server may still finish it until its process ends, which
it does before the block has ended.

What leaves a block. The task groups of the MCP client wrap whatever crosses them in an exception group, once per context
they unwind, and they cancel the caller when one of their own tasks fails. So the block hands on what happened as itself:
an error of the caller's own code as that error, a server that did not start and a transport that broke under the caller as
a ToolTransportError, and a cancellation that the caller's task was sent as a cancellation, wherever it landed. An error of
ending the processes after the caller's code has finished is dropped, since a server that still writes while it is being
ended would otherwise cost a finished run its result.
"""
from __future__ import annotations

import asyncio
import contextvars
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
import time
from pathlib import Path

from CausalDiagnosis.utils import model_request_retry as retry

SESSIONS = "one per server and episode"  # what a run folder records for a runner that plays on a block


TEARDOWN = "teardown"  # the second line of a return code file of a process that ended after the block began to close (failures.server_evidence)
HIDDEN_FOLDER = "hidden"  # the episode folder's part that no agent reads (run_episodes.HIDDEN)
_STARTED = contextvars.ContextVar("started_processes", default=None)  # [(server, process)] of the block being opened


def error_log(launcher, server):
    """The error log of the next start of a server whose launcher lies in an episode folder: hidden/<server>.<k>.stderr, k the
    number of earlier starts of that server there, so a restart writes a log of its own."""
    folder = Path(launcher).parent / HIDDEN_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    k = 0
    while (folder / f"{server}.{k}.stderr").exists():
        k += 1
    return folder / f"{server}.{k}.stderr"


def _record_processes():
    """Wraps the process start of the MCP client once, so that a block learns the processes it starts and can read their return
    codes after they end; a start outside a block is passed through unchanged."""
    from mcp.client import stdio

    original = stdio._create_platform_compatible_process
    if getattr(original, "_records_processes", False):
        return

    async def create(*args, **kwargs):
        process = await original(*args, **kwargs)
        started = _STARTED.get()
        if started is not None:
            started.append(process)
        return process
    create._records_processes = True
    stdio._create_platform_compatible_process = create


class ToolTransportError(RuntimeError):
    """A tool call without an answer, a server that did not start, or a transport that broke."""


class EpisodeSessions:
    """The sessions of an open block. ``tools`` holds what each server listed: {server name: its tools}."""

    def __init__(self, sessions, tools):
        self._sessions, self.tools = sessions, tools
        self.returncodes = {}  # {server: the return code of its process}, filled when the block has ended the processes
        self._open = True
        self._in_flight = set()  # the tasks that wait for an answer, which the end of the block gives up

    async def call(self, server, tool, arguments):
        """One tool call: (text, is_error), the answer of the tool and whether the server marked it as an error. A call has no
        time limit of its own: the deadline of the episode bounds it, and outside an episode it waits for its answer."""
        session = self._sessions.get(server)
        if session is None:
            raise ToolTransportError(f"{tool} of {server} got no answer: the block opened no server of that name")
        if not self._open:
            raise ToolTransportError(f"{tool} of {server} got no answer: the sessions are closed")
        at = retry.deadline_at(retry.current_deadline())  # the deadline of the episode bounds every call, whatever its own limit
        if at is not None and at - time.monotonic() <= 0:
            raise ToolTransportError(f"{tool} of {server} was not sent: the deadline of the episode has passed")
        task = asyncio.current_task()
        self._in_flight.add(task)
        answer = asyncio.ensure_future(session.call_tool(tool, arguments))
        try:
            # the deadline is read again at every expiry, since the call's own credit lands before its reply does
            for left in retry.waits_to_deadline():
                done, _pending = await asyncio.wait({answer}, timeout=left)
                if done:
                    break
            else:
                raise ToolTransportError(f"{tool} of {server} gave no answer before the deadline of the episode")
            if answer.cancelled():
                raise ToolTransportError(f"{tool} of {server} got no answer: the call was cancelled")
            result = answer.result()
        except asyncio.CancelledError:
            if self._open:
                raise
            task.uncancel()  # the cancellation was the block's own, sent by _close, and ends here
            raise ToolTransportError(f"{tool} of {server} got no answer: the sessions closed before it came") from None
        except ToolTransportError:
            raise
        except Exception as exc:  # noqa: BLE001 - whatever kept the answer away is a failure of the transport
            raise ToolTransportError(f"{tool} of {server} got no answer: {_named(exc)}") from exc
        finally:
            self._in_flight.discard(task)
            if not answer.done():
                answer.cancel()  # a call given up at the deadline or with the block, as the client cannot recall it
                await asyncio.wait({answer})
        return "\n".join(getattr(item, "text", str(item)) for item in result.content), bool(result.isError)

    def _close(self):
        """Gives up every call that another task still waits on, since the client does not answer a waiting request when
        its session closes, and returns those tasks."""
        self._open = False
        given_up = [task for task in self._in_flight if task is not asyncio.current_task()]
        for task in given_up:
            task.cancel()
        return given_up


def _named(error):
    """An error as ``TypeName: message``, an exception group as its errors in that form."""
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(_named(inner) for inner in error.exceptions)
    return f"{type(error).__name__}: {error}"


def _leaving(unwound, report, opening, finished):
    """What leaves a block whose unwinding raised ``unwound``, or None when nothing does. ``report`` is the error of the
    caller's code or of a server that did not start, ``opening`` the server that was being started, ``finished`` whether the
    caller's code ended without an error. The task is asked after the unwinding, when the cancellations that the client sent
    itself are taken back, so what is left was sent by the caller's side."""
    if report is not None and not isinstance(report, asyncio.CancelledError):
        return report
    if asyncio.current_task().cancelling():
        return report if report is not None else asyncio.CancelledError()
    if not isinstance(unwound, Exception):
        return unwound  # a keyboard interrupt or an exit of the interpreter, never dropped and never renamed
    if finished:
        return None
    what = f"the server {opening} did not start" if opening else "the sessions broke under the caller"
    error = ToolTransportError(f"{what}: {_named(unwound)}")
    error.__cause__ = unwound
    return error


def _waitid(pid, options):
    """os.waitid(P_PID, pid, options), or the C library's waitid where os lacks it (macOS before Python 3.13, whose siginfo
    holds si_pid at byte 12): None when the child has not changed state, its pid when it has."""
    if hasattr(os, "waitid"):
        result = os.waitid(os.P_PID, pid, options)
        return None if result is None else result.si_pid
    if sys.platform != "darwin":
        raise OSError(f"no waitid on {sys.platform}")
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    info = ctypes.create_string_buffer(128)
    if libc.waitid(ctypes.c_int(os.P_PID), ctypes.c_uint(pid), info, ctypes.c_int(options)) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))  # ECHILD comes back as ChildProcessError
    found = int.from_bytes(info.raw[12:16], sys.byteorder, signed=True)
    return found or None


def _running(process):
    """Whether a started process has not ended. asyncio sets the return code from its child watcher, a callback that can come
    after the end of a server has already failed the pending call, so the kernel is asked as well, without reaping the child
    (WNOWAIT). A child that a watcher thread has reaped already (ECHILD) has ended, and so has one the kernel cannot be asked
    about: its end then stays unmarked and reads as the server's."""
    if process.returncode is not None:
        return False
    try:
        return _waitid(process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
    except OSError:
        return False


def _running_ids(processes):
    """The ids of the processes still running, an empty set when the query fails, so that the close after it always runs."""
    try:
        return {id(process) for _name, process, _log in processes if _running(process)}
    except Exception:
        return set()


@asynccontextmanager
async def open_sessions(server_paths):
    """The block of one episode: an EpisodeSessions over one process per server of ``{server name: launcher path}``,
    started in that order."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _record_processes()
    report = opening = leaving = opened = None
    finished, given_up = False, []
    started, processes = _STARTED.set([]), []
    alive_at_close = set()  # the processes still running when the block began to close, so their end came after what the agent saw
    try:
        async with AsyncExitStack() as stack:
            sessions, tools = {}, {}
            for name, path in server_paths.items():
                opening = name
                try:
                    log_path = error_log(path, name)
                    log = stack.enter_context(open(log_path, "w", encoding="utf-8"))
                    read, write = await stack.enter_async_context(stdio_client(StdioServerParameters(command="python", args=["-I", str(path)]),
                                                                               errlog=log))
                    processes.append((name, _STARTED.get()[-1], log_path))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    # listed for every agent at the opening: the client asks for the list at the first call of a tool it does
                    # not know, so without this the first calls of one agent would carry a request that another's do not
                    sessions[name], tools[name] = session, (await session.list_tools()).tools
                except Exception as exc:  # noqa: BLE001 - whatever the client raised for a server that did not start
                    report = ToolTransportError(f"the server {name} did not start: {_named(exc)}")
                    raise report from exc
            opening = None
            opened = EpisodeSessions(sessions, tools)
            try:
                yield opened
                finished = True
            except BaseException as exc:
                report = exc
                raise
            finally:
                alive_at_close = _running_ids(processes)
                given_up = opened._close()
    except BaseException as unwound:
        leaving = _leaving(unwound, report, opening, finished)
    finally:
        _STARTED.reset(started)
    for _name, process, log_path in processes:  # every process has ended with the stack, so its return code is known
        by_teardown = id(process) in alive_at_close  # any end after the close, a signal of the client or an exit of its own
        log_path.with_suffix(".returncode").write_text(f"{process.returncode}\n" + (f"{TEARDOWN}\n" if by_teardown else ""),
                                                       encoding="utf-8")
    if opened is not None:
        opened.returncodes = {name: process.returncode for name, process, _log in processes}
    if given_up:
        await asyncio.wait(given_up)  # each has raised by now or does so at once, so its caller reads the error before the block has ended
    if leaving is not None:
        raise leaving  # outside the handler, so that an error of the caller keeps the context it came with
