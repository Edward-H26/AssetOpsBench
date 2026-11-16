"""Stand-in MCP servers for the Agent-As-Tool port's tests, one per server name of the AssetOpsBench rows, each with a few
tools answered from constants. Every call appends one JSON line (server, tool, arguments) to the file the launcher names, so
a test can tell which tool ran with which arguments. The task server's finish ends the episode: every later call of any
server is answered with the refusal the rig's servers give after the end."""
import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

EPISODE_OVER = "the episode is over; this call changed nothing"
# the example start of a run over the stand-in task server, which fits its submit_diagnosis(alpha, beta, evidence_ids)
EXAMPLE_START = {"alpha": 4, "beta": 7}


def serve(name, log_path):
    log, over = Path(log_path), Path(log_path).with_suffix(".over")
    mcp = FastMCP(f"stub-{name}")

    def record(tool, **arguments):
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"server": name, "tool": tool, "arguments": arguments}) + "\n")
        return {"error": EPISODE_OVER} if over.exists() else None

    if name == "iot":
        @mcp.tool()
        def sites() -> dict:
            """List the IoT sites."""
            return record("sites") or {"sites": ["MAIN"]}

        @mcp.tool()
        def assets(site_name: str) -> dict:
            """List the assets of a site."""
            return record("assets", site_name=site_name) or {"site_name": site_name, "assets": ["CHL-A100"]}

        @mcp.tool()
        def history(site_name: str, asset_id: str, limit: int = 5) -> dict:
            """Return a page of the asset's telemetry."""
            return record("history", site_name=site_name, asset_id=asset_id, limit=limit) or {"observations": [], "limit": limit}
    elif name == "task":
        @mcp.tool()
        def get_asset_profile() -> dict:
            """The asset profile."""
            return record("get_asset_profile") or {"asset": {"asset_id": "CHL-A100"}}

        @mcp.tool()
        def submit_diagnosis(alpha: int, beta: int, evidence_ids: str) -> dict:
            """Record a diagnosis of the components alpha and beta with the cited evidence."""
            return record("submit_diagnosis", alpha=alpha, beta=beta, evidence_ids=evidence_ids) or {
                "diagnosis": {"alpha": alpha, "beta": beta}, "evidence_ids": [e for e in evidence_ids.split(",") if e]}

        @mcp.tool()
        def finish(summary: str) -> dict:
            """End the episode with a summary."""
            refused = record("finish", summary=summary)
            over.write_text("over", encoding="utf-8")
            return refused or {"finished": True}
    elif name == "fmsr":
        @mcp.tool()
        def get_failure_modes(asset_class: str) -> dict:
            """Read the failure modes of an asset class."""
            return record("get_failure_modes", asset_class=asset_class) or {"failure_modes": []}
    elif name == "wo":
        @mcp.tool()
        def list_workorders(site_id: str = "MAIN") -> dict:
            """List the work orders of a site."""
            return record("list_workorders", site_id=site_id) or {"workorders": []}
    elif name == "causal_diagnosis":
        @mcp.tool()
        def rank_root_causes(top_k: int) -> dict:
            """Rank the candidate root causes."""
            return record("rank_root_causes", top_k=top_k) or {"candidates": []}
    elif name == "utilities":
        @mcp.tool()
        def current_date_time() -> dict:
            """The current date and time."""
            return record("current_date_time") or {"currentDateTime": "2020-07-01T00:00:00+00:00"}
    elif name == "tsfm":
        @mcp.tool()
        def get_ai_tasks() -> dict:
            """The AI tasks the server supports."""
            return record("get_ai_tasks") or {"tasks": []}
    mcp.run(transport="stdio")


LAUNCHER = '''import sys
sys.dont_write_bytecode = True
sys.path.insert(0, {folder!r})
from stub_servers import serve
serve({name!r}, {log!r})
'''


def write_launchers(folder, names):
    """{name: launcher path} for the stand-in servers, all logging into folder/calls.jsonl."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "calls.jsonl"
    paths = {}
    for name in names:
        path = folder / f"{name}_server.py"
        path.write_text(LAUNCHER.format(folder=str(Path(__file__).resolve().parent), name=name, log=str(log)), encoding="utf-8")
        paths[name] = path
    return paths


def calls(folder):
    log = Path(folder) / "calls.jsonl"
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


if __name__ == "__main__":
    serve(sys.argv[1], sys.argv[2])
