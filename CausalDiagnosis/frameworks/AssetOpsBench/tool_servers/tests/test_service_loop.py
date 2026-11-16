"""The service thread and the hold of a server process (servers/service_loop.py, CausalDiagnosis/mcp_servers/utils/transcript_hooks.py; the clock of the
thinking time in CausalDiagnosis/docs/environment.md): every call of the meter runs on the one long-lived service thread in the
caller's context, a cancelled caller cancels its call there, and the hold lets no other request of the process work while a
call is served, a listing flood included."""
import asyncio
import threading

import mcp.types as types
from mcp.server.lowlevel import Server

from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.mcp_servers.utils.service_loop import ServiceLoop
from CausalDiagnosis.mcp_servers.utils.transcript_hooks import transcribed


async def test_a_call_runs_on_the_service_thread_in_the_callers_context():
    service = ServiceLoop()

    async def where():
        return threading.current_thread().name, transcript._CALL.get()

    with transcript.serving({"x": 1}) as call:
        name, seen = await service.run(where())
    assert name == "service" and seen is call  # a note made on the service thread reaches the wrapper's line


async def test_a_cancelled_caller_cancels_its_call_on_the_service_loop():
    service, started, cancelled = ServiceLoop(), threading.Event(), threading.Event()

    async def long():
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.ensure_future(service.run(long()))
    while not started.is_set():
        await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    for _ in range(100):
        if cancelled.is_set():
            break
        await asyncio.sleep(0.01)
    assert cancelled.is_set()


async def test_the_hold_serves_no_listing_while_a_call_is_served(tmp_path):
    """A slow call and a flood of listings sent at once: every listing line follows the call's line, since the hold keeps the
    listings waiting until the call's answer is complete."""
    app, release = Server("probe"), asyncio.Event()

    @app.list_tools()
    async def listing():
        return [types.Tool(name="slow", description="", inputSchema={"type": "object"})]

    @app.call_tool(validate_input=False)
    async def slow(name, arguments):
        transcript.append(tmp_path, "inside", server="probe")
        await release.wait()
        return [types.TextContent(type="text", text="done")]

    transcript.begin(tmp_path)
    handlers = transcribed(app, tmp_path, "probe").request_handlers
    call = asyncio.ensure_future(handlers[types.CallToolRequest](
        types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(name="slow", arguments={}))))
    await asyncio.sleep(0.05)
    floods = [asyncio.ensure_future(handlers[types.ListToolsRequest](types.ListToolsRequest(method="tools/list"))) for _ in range(20)]
    await asyncio.sleep(0.05)
    release.set()
    await asyncio.gather(call, *floods)
    kinds = [line["kind"] for line in transcript.read(tmp_path)][1:]
    assert kinds == ["inside", "call"] + ["list"] * 20


async def test_the_first_call_of_a_process_completes_under_the_hold(tmp_path):
    """The MCP package fills its tool cache on the first call of a process by calling the listing handler from inside the
    call handler; the hold must not make that inner listing wait for the call that holds it, or the call never ends."""
    app = Server("probe")

    @app.list_tools()
    async def listing():
        return [types.Tool(name="one", description="", inputSchema={"type": "object"})]

    @app.call_tool(validate_input=False)
    async def one(name, arguments):
        return [types.TextContent(type="text", text="done")]

    transcript.begin(tmp_path)
    handlers = transcribed(app, tmp_path, "probe").request_handlers
    request = types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(name="one", arguments={}))
    result = await asyncio.wait_for(handlers[types.CallToolRequest](request), timeout=10)
    assert not result.root.isError
    assert [line["kind"] for line in transcript.read(tmp_path)][1:] == ["call"]
