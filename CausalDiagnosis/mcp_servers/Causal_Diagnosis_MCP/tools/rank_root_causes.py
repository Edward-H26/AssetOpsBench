import logging


COMPACT_LEFT_OUT = ("attribution", "level_probability", "stability", "monte_carlo")  # the keys that the default reply leaves out
TIMING = logging.getLogger("CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.timing")
if not TIMING.handlers:  # its own handler, since FastMCP logs warnings alone and a ranking's time is no warning
    TIMING.addHandler(logging.StreamHandler())
    TIMING.setLevel(logging.INFO)
    TIMING.propagate = False
