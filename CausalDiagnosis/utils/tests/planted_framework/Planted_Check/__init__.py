"""The declaration of a framework planted by test_planted_framework.py: two agents, a judge, a tool surface on its own server
names over the launch module of AssetOpsBench's servers, a run kind, a profile check, a digest family, a command and lists.
Every value is a literal; code is named by "module:attribute" in planted.py beside the test."""

AGENTS = {
    "planted_model": {"route": "chat", "surfaces": ("planted",), "trained": False, "paper_paradigm": False,
                      "entry": "CausalDiagnosis.utils.tests.planted_framework.planted:AGENT"},
    "planted_check": {"model_agent": False, "surfaces": ("planted",), "trained": False, "paper_paradigm": False,
                      "entry": "CausalDiagnosis.utils.tests.planted_framework.planted:AGENT"},
}

SURFACES = {
    "planted": {"servers": ("planted_iot", "planted_task"), "state_file": "episode.json",
                "launch_module": "CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server",
                "stores": ("memory",), "store_environment": (), "export_file": "planted_documents.json",
                "default_on_row_presets": False, "entry": "CausalDiagnosis.utils.tests.planted_framework.planted:SURFACE"},
}

JUDGES = {"planted_judge": {"protocols": ("single",), "entry": "CausalDiagnosis.utils.tests.planted_framework.planted:JUDGE"}}

RUN_KINDS = {"planted_run": {"entry": "CausalDiagnosis.utils.tests.planted_framework.planted:RUN_KIND"}}

PROFILE_CHECKS = {"planted": "CausalDiagnosis.utils.tests.planted_framework.planted:profile_problems"}

DIGEST_FAMILIES = {"planted_rows": "CausalDiagnosis.utils.tests.planted_framework.planted:FAMILY"}

COMMANDS = {"planted-command": "CausalDiagnosis.utils.tests.planted_framework.planted"}

LISTS = {"agent:planted_model": ("agents/",)}
