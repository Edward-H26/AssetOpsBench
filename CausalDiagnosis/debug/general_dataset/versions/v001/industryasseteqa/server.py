"""Expose the public task, a fact read and an answer submission over MCP."""

import json

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from CausalDiagnosis.mcp_servers.utils.transcript_hooks import transcribed


def build_server(episode, folder, *, normalized=False):
    """Bind a transcribed server to one question without evaluator references."""
    app = FastMCP("maintenance_qa")

    @app.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_task() -> dict:
        """Read the question. Read its fact before submitting an answer.

        Submission accepts direct_answer and reasoning_answer as nonempty strings,
        provenance with fact_id and a nonempty features list, and confidence from
        zero to one. Source file and row may also be cited. Accepted submission
        ends the episode. Structural and provenance checks do not judge quality.
        """
        return {"result": episode.task} if normalized else episode.task

    @app.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def read_fact() -> dict:
        """Read the measurements and source metadata for this question."""
        fact = episode.read_fact()
        if not normalized:
            return fact
        return {"result": fact, "observations": [{"id": f"fact/{fact['fact_id']}",
                                                   "text": json.dumps(fact, allow_nan=False)}]}

    @app.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def submit_answer(answer: dict) -> dict:
        """Submit a structured answer after reading the fact, ending on acceptance."""
        result = episode.submit_answer(answer)
        if not normalized:
            return result
        if result.get("accepted") is not True:
            return {"refused": "submission failed structure or provenance checks", "checks": result}
        return {"result": result, "status": {"over": True}}

    return transcribed(app, folder, "maintenance_qa", keep_text=True)
