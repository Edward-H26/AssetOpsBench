"""Expose a scripted episode with its public tool schemas over stdio MCP."""

import asyncio
import json
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


def run_server(episode, folder):
    app = Server("scripted_case")
    folder = Path(folder)
    private = folder / "hidden"
    private.mkdir(exist_ok=True)

    def checkpoint():
        path = private / "replay.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"status": episode.status(), "trace": episode.trace,
                                         "submission": episode.submission}, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    @app.list_tools()
    async def listing():
        return [Tool(name=item["name"], description=item["description"], inputSchema=item["schema"])
                for item in episode.tools.values()]

    @app.call_tool(validate_input=False)
    async def call(name, arguments):
        result = episode.call(name, arguments or {})
        checkpoint()
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, allow_nan=False))]

    async def serve():
        checkpoint()
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())

    asyncio.run(serve())
