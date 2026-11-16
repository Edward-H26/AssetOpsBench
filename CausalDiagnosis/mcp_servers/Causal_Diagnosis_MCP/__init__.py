"""The diagnostics MCP server: four read only tools over the evidence an episode has released (CausalDiagnosis/docs/causal-diagnosis-mcp.md).

The server holds what the agent reasons with; the action server of CausalDiagnosis/mcp_servers/Environment_Actions_MCP holds what changes the
rig. One server process per episode, bound to the episode folder by a launcher file, as for the action server. The fits of
an episode leave out the block of its start, which the launcher names and nothing else uses: every
hypothesis is scored without that block and without its own, from the cycles and the pair selections of models.json, and the
asset enters through asset_view.py alone; no tool opens a data file, calls a language model, or reads a hidden label.
"""
