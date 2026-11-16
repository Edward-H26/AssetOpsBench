"""A block starts each server isolated (python -I), gives each start its own error log under
hidden/ of the launcher's folder, and records the return code of every process it started."""
import asyncio
from pathlib import Path

from CausalDiagnosis.agents.utils.mcp_sessions import ToolTransportError, open_sessions

STAND_IN = '''
import os
import sys
from pathlib import Path

Path(FOLDER, NAME + ".isolated").write_text(str(sys.flags.isolated), encoding="utf-8")
print("started " + NAME, file=sys.stderr, flush=True)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stand-in-" + NAME, log_level="ERROR")


@mcp.tool()
def ping() -> dict:
    """Answer."""
    return {"ok": True}


@mcp.tool()
def die() -> dict:
    """End the process without an answer."""
    os._exit(3)


mcp.run(transport="stdio")
'''


def stand_in(folder, name):
    path = Path(folder) / f"{name}_server.py"
    path.write_text(f"NAME, FOLDER = {name!r}, {str(folder)!r}\n{STAND_IN}", encoding="utf-8")
    return path


def test_each_server_runs_isolated_and_writes_its_own_error_log_per_start(tmp_path):
    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a")}) as sessions:
            await sessions.call("a", "ping", {})
        return sessions

    first = asyncio.run(play())
    asyncio.run(play())
    assert (tmp_path / "a.isolated").read_text(encoding="utf-8") == "1"
    logs = sorted(p.name for p in (tmp_path / "hidden").iterdir())
    assert logs == ["a.0.returncode", "a.0.stderr", "a.1.returncode", "a.1.stderr"]
    assert "started a" in (tmp_path / "hidden" / "a.0.stderr").read_text(encoding="utf-8")
    assert set(first.returncodes) == {"a"} and first.returncodes["a"] is not None


def test_the_return_code_of_a_server_that_ended_itself_is_recorded(tmp_path):
    holder = {}

    async def play():
        async with open_sessions({"a": stand_in(tmp_path, "a"), "b": stand_in(tmp_path, "b")}) as sessions:
            holder["sessions"] = sessions
            try:
                await sessions.call("a", "die", {})
            except ToolTransportError:
                pass
            await sessions.call("b", "ping", {})

    try:
        asyncio.run(play())
    except ToolTransportError:
        pass  # a server that ended under the caller may break the block, which still records the codes
    assert holder["sessions"].returncodes.get("a") == 3
    assert (tmp_path / "hidden" / "a.0.returncode").read_text(encoding="utf-8") == "3\n"
