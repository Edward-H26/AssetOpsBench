"""The diagnostics MCP server: four read only tools over the evidence an episode has released (docs/hydraulic_rig/causal-diagnosis-server.md).

The server holds what the agent reasons with; the action server of src/servers/hydraulic_rig holds what changes the
rig. One server process per episode, bound to the episode folder by a launcher file, as for the action server. Every
model inside the tools is fitted once on a fixed pool of combinations by fit.py and read from models.json; no tool opens a
data file, calls a language model, or reads a hidden label.
"""
