"""How every agent of the fork reads the answer of a tool: the JSON object an answer holds, and the start of the refusal
every server answers after the end of an episode. The module imports nothing of the fork, so the controller of the
CausalDiagnosis agent and the agents of a framework read the same answers by the same rules.
"""
import json

EPISODE_OVER = "the episode is over"  # the start of the refusal every server answers after the end (CausalDiagnosis/mcp_servers/Environment_Actions_MCP/episode_state.py)


def json_object(text):
    """The JSON object a text holds, or None."""
    try:
        value = json.loads(text) if text else None
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None
