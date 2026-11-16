"""The ready line of a server process (CausalDiagnosis/docs/environment.md, the clock of the thinking time).

Each server main writes one line of kind ready to the episode's transcript immediately before its stdio loop, so no request
can arrive before it. The line carries the launcher's stamp s (its first statement), the reading r taken here, both on the wall
clock, the monotonic clock and the main thread's thread time, and the facts of the process that decide whether its readings
can be trusted: the interpreter flags, a trace or profile function, tracing of allocations, a customize module, the
environment variables that slow the interpreter, the thread count and the SHA-256 of the launcher. The first start of each
server is credited from these readings (servers/thinking_clock.py), and a counted call of a process whose line shows a
problem is refused (problem)."""
import hashlib
import os
import sys
import threading
import time
import tracemalloc
from pathlib import Path

from CausalDiagnosis.mcp_servers.utils import transcript

# the environment variables that slow the interpreter or trace it, with the values that do
SLOWING_VARIABLES = {"PYTHONMALLOC": ("debug", "malloc_debug", "pymalloc_debug"), "PYTHONASYNCIODEBUG": None, "PYTHONTRACEMALLOC": None}
FLAG_NAMES = ("isolated", "dev_mode", "no_site", "ignore_environment", "optimize", "safe_path")


def _launcher_sha256():
    """The SHA-256 of the file that started the process, None for a process that no file started."""
    path = getattr(sys.modules.get("__main__"), "__file__", None)
    return transcript.file_sha256(path) if path else None


def facts():
    """The facts of this process that the ready line records."""
    return {"flags": {name: getattr(sys.flags, name) for name in FLAG_NAMES}, "xoptions": sorted(map(str, sys._xoptions)),
            "tracemalloc": tracemalloc.is_tracing(), "trace": sys.gettrace() is not None, "profile": sys.getprofile() is not None,
            "threads": threading.active_count(), "customize": sorted(m for m in ("sitecustomize", "usercustomize") if m in sys.modules),
            "executable": sys.executable, "launcher_sha256": _launcher_sha256(),
            "environment": {name: os.environ[name] for name in SLOWING_VARIABLES if name in os.environ}}


def announce(folder, server, started=None):
    """Write the ready line of this process: s from the launcher's stamp (time.time(), time.monotonic(), time.thread_time()),
    None without a launcher, and r read now, just before the stdio loop. Returns the line number, None without a folder."""
    s_wall, s_mono, s_thread = started if started is not None else (None, None, None)
    return transcript.append(folder, "ready", server=server, s_wall=s_wall, s_mono=s_mono, s_thread_cpu=s_thread,
                             r_wall=time.time(), r_mono=time.monotonic(), r_thread_cpu=time.thread_time(), **facts())


def problem(line, launcher_sha256=None, threads=None):
    """Why the readings of a process cannot be trusted, from its ready line, None when they can: a process that is not
    isolated, a trace or profile function, tracing of allocations, -X dev or importtime, a customize module, an environment
    variable that slows the interpreter, a launcher other than the one the driver wrote (launcher_sha256 where the driver
    recorded it), or more threads than the baseline of its kind (threads where the driver measured it)."""
    flags = line.get("flags") or {}
    if not flags.get("isolated"):
        return "the server process was not started isolated (python -I)"
    if line.get("trace") or line.get("profile"):
        return "the server process runs a trace or profile function"
    if line.get("tracemalloc"):
        return "the server process traces its allocations"
    if flags.get("dev_mode") or any(o.split("=")[0] in ("dev", "importtime", "tracemalloc") for o in line.get("xoptions") or ()):
        return "the server process runs with -X dev, importtime or tracemalloc"
    if line.get("customize"):
        return f"the server process loaded {', '.join(line['customize'])}"
    for name, value in (line.get("environment") or {}).items():
        if SLOWING_VARIABLES.get(name) is None or value in SLOWING_VARIABLES[name]:
            return f"the server process runs with {name}={value}"
    if launcher_sha256 is not None and line.get("launcher_sha256") != launcher_sha256:
        return "the server process was started from a launcher other than the one the driver wrote"
    if threads is not None and (line.get("threads") or 0) > threads:
        return f"the server process ran {line.get('threads')} threads at its start, above the {threads} of its kind"
    return None
