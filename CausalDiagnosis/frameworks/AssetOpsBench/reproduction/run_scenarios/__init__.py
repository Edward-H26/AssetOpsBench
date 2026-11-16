"""Play the two paper-era agents of AssetOpsBench on the paper's scenarios (CausalDiagnosis/frameworks/AssetOpsBench/reproduction/README.md).

usage, from the fork root with the fork's virtual environment first on PATH (every tool server starts as python):
  python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.run_scenarios --scenarios FILE --out DIR [--set runnable|pilot|recovered | --ids ID [ID ...]]
      --layer-b DIR [--agents agent_as_tool_paper plan_execute_paper] [--repeats 1] [--model NAME] [--base-url URL]
      [--episode-timeout 1800] [--no-reload] [--server NAME=SCRIPT ...] [--allow-unpinned]

agent_as_tool_paper is the paper-era Agent-As-Tool of CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper, IBM's ReAct supervisor over the paper's five
agents and JSONReader by raw text completions, whose steps are its supervisor's ReAct counter, the paper's own count.
plan_execute_paper is the paper-era Plan-Execute of CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper, AgentHive's planner, reviewer and
summarizer over the same agents, whose steps are its executed plan steps. Both play by default, and acceptance_check.py compares
the runs of each with the paper. The servers and the data are those of IBM's public branches, which fetch_public_servers_and_data.py fetches into
--layer-b. They reach the model at --base-url (default BASE_URL, which serve_model.sbatch exports) under the name --model
(default SERVED_NAME) with the key in API_KEY, and they see the six MCP servers of main-0.x under the names of
DEFAULT_SERVER_PATHS of src/agent/runner.py. Every server starts from a launcher file in a private temporary folder, since
a stdio server inherits only HOME, LOGNAME, PATH, SHELL, TERM and USER from its client: the launcher sets the CouchDB
settings of this process and points the FMSR server's generate tools at the agent's own model and endpoint. Before every
episode the CouchDB databases are dropped and that layer's data is loaded again, as the fork's scenario suite runner does
with its manifest, so that no episode reads what an earlier one wrote (--no-reload skips this, for tests and dry runs).

Episodes run in repeat order, every scenario with every agent before the next repeat, so an interrupted run still holds
whole repeats. Each episode folder holds episode.json (question, answer, error, runtime and step counts), llm_calls.json
(every model call) and the trajectory record that the runner persists and the fork's Evaluator reads. The run folder holds
run.json with the model, the endpoint, the scenario list, the repeats, the scenario file's hash and the fork's git
describe. Nothing is scored here, since score_runs.py scores the folders. The functions that count the steps of a run of
AssetOpsBench's released runners from its model call log stay here for the rig's tables_from_runs/.
"""

from CausalDiagnosis.frameworks.AssetOpsBench.agents import upstream_agent_package  # noqa: E402,F401  registers upstream's agent package by its path

from . import server_launchers
from . import step_counts
from . import play_episode
from . import run_record
from . import command_line

from .server_launchers import parse_servers, reload_couchdb, server_env, write_launchers
from .play_episode import Settings, agent_runs, pe_port_runner, play, timed_run
from .run_record import CODE
from .command_line import main, parse_args
