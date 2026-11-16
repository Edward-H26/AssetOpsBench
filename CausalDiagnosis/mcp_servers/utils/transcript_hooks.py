"""The transcript of an episode at the two handlers through which the MCP package serves a client: the tool listing and the
tool call (servers/transcript.py).

A server of an episode is wrapped once, when it is built. The wrapper sees what goes back to the client, after the package
has turned the result of a tool into text and after its own checks, so the text that is hashed is the text the client
receives, also for a call that failed before the tool ran. A call whose line the server wrote itself, inside the lock of the
episode where its clock reading is known, gets no second line here.

The wrapper also holds the hold of the process: one lock from the moment a listing or a call reaches its handler until its
answer is complete, so no other request of the process works while a counted call is served, and the thread time of that
service is the call's alone (CausalDiagnosis/docs/environment.md, the clock of the thinking time).
"""
import asyncio
import weakref

import mcp.types as types

from CausalDiagnosis.mcp_servers.utils import transcript


def lowlevel_of(app):
    """The low level server of an app. A FastMCP keeps it under one private name, and a low level server is its own."""
    return getattr(app, "_mcp_server", app)


def sent_text(result):
    """The text of a tool result as every runner joins it: the text of each content block, one per line."""
    return "\n".join(getattr(block, "text", str(block)) for block in result.content)


def transcribed(app, folder, server, keep_text=False):
    """Wrap the listing handler and the call handler of an app so that the episode folder's transcript holds every listing
    and every call of this server process. keep_text keeps the answer text of each call in its line, for a server that only
    reads the released evidence. Without a folder the app is returned as it is, as for a listing before any episode."""
    if folder is None:
        return app
    handlers = lowlevel_of(app).request_handlers
    listing, calling = handlers[types.ListToolsRequest], handlers[types.CallToolRequest]
    # the hold of the process, one per event loop, made on the loop that first serves a request; keyed by the loop itself and
    # dropped with it, so a later loop that reuses a collected loop's id never inherits a lock that loop left taken
    holds = weakref.WeakKeyDictionary()

    def hold():
        return holds.setdefault(asyncio.get_running_loop(), asyncio.Lock())

    async def list_tools(request):
        if request is None:  # the package fills its tool cache from inside a call that already holds the hold
            return await _list_tools(request)
        async with hold():
            return await _list_tools(request)

    async def call_tool(request):
        async with hold():
            return await _call_tool(request)

    async def _list_tools(request):
        result = await listing(request)
        tools = getattr(result.root, "tools", None)
        # the package calls this handler without a request to fill its own tool cache before the first call of a process,
        # which is no listing that a client asked for
        if request is not None and tools is not None:
            transcript.append(folder, "list", server=server, listing_sha256=transcript.listing_sha256(tools))
        return result

    async def _call_tool(request):
        with transcript.serving(request.params.arguments or {}) as call:
            result = await calling(request)
            answer = result.root
            if not call["recorded"] and isinstance(answer, types.CallToolResult):
                transcript.answered(folder, server, request.params.name, call["arguments"], sent_text(answer), answer.isError,
                                    keep_text=keep_text, **call["fields"])
        return result

    handlers[types.ListToolsRequest] = list_tools
    handlers[types.CallToolRequest] = call_tool
    return app
