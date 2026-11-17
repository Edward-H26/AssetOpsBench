"""The declaration of AssetOpsBench: what this framework brings to the core, read by CausalDiagnosis/utils/frameworks.py as
data and never imported (frameworks/README.md). Every value is a plain literal; code is named by "module:attribute" and
imported only by a run that asks for it.

Modules of this package, one per line:

LLM_Judges
agents
reproduction
scenarios
tool_servers
"""

# The agent kinds. model_agent: the agent calls a model (an agent without the flag counts as one, so the serving check holds for
# it); route: the route of its model calls; surfaces: the tool surfaces it plays; trained: it
# enters the training tables; paper_paradigm: it is a paradigm of the paper's tables; modules: the modules through which the
# driver reaches its code, which the test of the agent files imports alone in a fresh interpreter.
AGENTS = {
    "plan_execute": {"model_agent": True, "route": "completion", "surfaces": ("aob",), "trained": True, "paper_paradigm": True,
                     "modules": ("CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper",),
                     "entry": "CausalDiagnosis.frameworks.AssetOpsBench.agents.agent_runs:PLAN_EXECUTE"},
    "agent_as_tool": {"model_agent": True, "route": "completion", "surfaces": ("aob",), "trained": True, "paper_paradigm": True,
                      "modules": ("CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper",),
                      "entry": "CausalDiagnosis.frameworks.AssetOpsBench.agents.agent_runs:AGENT_AS_TOOL"},
    "scripted": {"model_agent": False, "surfaces": ("action", "aob"), "trained": False, "paper_paradigm": False,
                 "modules": ("CausalDiagnosis.frameworks.AssetOpsBench.agents.scripted_wiring_check",),
                 "entry": "CausalDiagnosis.frameworks.AssetOpsBench.agents.scripted_wiring_check:SCRIPTED"},
}

# The tool surfaces. servers: the server names in the order the driver registers them; state_file: the episode file every
# server of the surface reads; launch_module: the module a server process runs; stores: the document stores the servers serve
# from, the first the default; store_environment: the variables a store other than the first reads; export_file: the file
# the documents of an episode are exported to; default_on_row_presets: the surface is the default of --servers on the presets
# of the table rows; action_server_name: the server that serves the environment's actions on this surface; covers: the environment's actions that tools of
# the framework serve; withheld_actions: the environment's actions the surface never serves; levels: the levels it runs at
# (utils/frameworks.py LEVELS); gateway: the code that serves the environment's actions on the action server.
SURFACES = {
    "aob": {"servers": ("iot", "fmsr", "wo", "task"), "state_file": "episode.json",
            "launch_module": "CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server",
            "stores": ("memory", "couchdb"), "store_environment": ("COUCHDB_URL", "COUCHDB_USERNAME", "COUCHDB_PASSWORD"),
            "export_file": "aob_documents.json", "default_on_row_presets": True,
            "action_server_name": "task",
            "covers": {"read_sensors": ("iot.history", "iot.latest_reading"), "get_failure_modes": ("fmsr.get_failure_modes",),
                       "create_work_order": ("wo.generate_work_order",), "get_fix_history": ("wo.list_workorders",)},
            "withheld_actions": (), "levels": ("framework", "full"),
            "gateway": "CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer:task_gateway",
            "entry": "CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tool_surface:AOB"},
}

# The judges, with the protocols each runs.
JUDGES = {
    "local": {"protocols": ("single", "paper"),
              "entry": "CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.episode_judge:JUDGE"},
}

# The scenario sets a preset may anchor, with the asset whose scenarios they are and the variable that names a local file.
SCENARIO_SETS = {
    "anchored": {"asset_id": "hyd_1", "env": "HYD_SCENARIOS_PATH",
                 "entry": "CausalDiagnosis.frameworks.AssetOpsBench.scenarios.scenario_anchors:ANCHORED"},
}

# The kinds of run the ledger records beside the gym runs.
RUN_KINDS = {
    "aob_repro": {"entry": "CausalDiagnosis.frameworks.AssetOpsBench.reproduction.ledger_fields:RUN_KIND"},
}

# The folders of presets under this folder, read beside CausalDiagnosis/configs/.
CONFIGS = ("scenarios",)

# The variables of the cluster jobs that the ledger records for the runs of this framework.
JOB_ENVIRONMENT = ("COUCHDB_IMAGE_SHA256",)

# The checks a surface runs on an asset profile when a run of it starts, and the digest families of the surface.
PROFILE_CHECKS = {"aob": "CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.profile_documents:surface_profile_problems"}
DIGEST_FAMILIES = {"rows": "CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.digest_families:ROWS"}

# The commands of the group assetopsbench (python -m CausalDiagnosis assetopsbench <command>), each the module it starts.
COMMANDS = {
    "episode-judge": "CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.episode_judge",
    "judge-finished-runs": "CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.judge_finished_runs",
    "download-scenarios": "CausalDiagnosis.frameworks.AssetOpsBench.scenarios.download_scenarios",
    "latency-probe": "CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.latency_probe",
    "run-scenarios": "CausalDiagnosis.frameworks.AssetOpsBench.reproduction.run_scenarios",
    "score-runs": "CausalDiagnosis.frameworks.AssetOpsBench.reproduction.score_runs",
    "fetch-public-servers-and-data": "CausalDiagnosis.frameworks.AssetOpsBench.reproduction.fetch_public_servers_and_data",
    "acceptance-check": "CausalDiagnosis.frameworks.AssetOpsBench.reproduction.acceptance_check",
}

# The identity lists (experiment_tables.lists_of), paths relative to this folder: the core list each path lies in (a name of
# experiment_tables.CODE_LISTS), or agent:<name> for an agent of this framework alone; the first list that matches wins. README.md
# and LICENSE fall to the core list documents, tests/ to checks.
LISTS = {
    "harness": ("__init__.py", "agents/__init__.py", "LLM_Judges/__init__.py", "reproduction/__init__.py", "cluster_jobs/"),
    "analysis": ("tool_servers/digest_families.py",),
    "episode": ("tool_servers/", "scenarios/", "agents/causal_diagnosis_agent_surface.py"),
    "scoring": ("LLM_Judges/",),
    "reproduction": ("reproduction/",),
    "agent:plan_execute": ("agents/Plan_Execute_Paper/", "agents/Agent_As_Tool_Paper/", "agents/ReActXen/", "agents/agent_runs.py",
                           "agents/upstream_agent_package.py"),
    "agent:agent_as_tool": ("agents/Agent_As_Tool_Paper/", "agents/ReActXen/", "agents/agent_runs.py", "agents/upstream_agent_package.py"),
    "agent:scripted": ("agents/scripted_wiring_check.py", "agents/upstream_agent_package.py"),
}
