"""pytest does not collect this folder: the debug overlay replaces the agent's prompts module and the planning method of its
controller for the whole process, and the frozen copies under versions/ and rounds/ repeat the name test_protocol.py. Each
test_protocol.py runs on its own with python, as maintenance_loop/README.md says."""
collect_ignore = ["maintenance_loop"]
