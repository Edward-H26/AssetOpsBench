"""A stand-in MCP server for the driver tests: two IoT tools over FastMCP answered from constants. Each answer carries the
CouchDB address the server process sees, so a test can tell that the launcher handed over the settings a stdio server does
not inherit from its client."""
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("aob-stub")


@mcp.tool()
def sites() -> dict:
    """List the IoT sites."""
    return {"sites": ["MAIN"], "couchdb_url": os.environ.get("COUCHDB_URL")}


@mcp.tool()
def assets(site_name: str) -> dict:
    """List the assets of a site."""
    return {"site_name": site_name, "assets": ["Chiller 6", "Chiller 9"], "couchdb_url": os.environ.get("COUCHDB_URL")}


if __name__ == "__main__":
    mcp.run(transport="stdio")
