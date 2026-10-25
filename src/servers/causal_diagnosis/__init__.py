"""The causal-diagnosis MCP server: seven tools over the evidence an episode has released (docs/hydraulic_rig/causal-diagnosis-server.md).

The server holds what the agent reasons with; the environment-action server of src/servers/hydraulic_rig holds what
changes the world. One server process per episode, bound to the episode folder by a launcher file, as for the
environment-action server. Every model inside the tools is fitted once on a fixed pool of combinations by fit.py and
read from models.json; no tool opens a data file, calls a language model, or reads a hidden label.
"""
