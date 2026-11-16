from pathlib import Path


# the paper-era Agent-As-Tool of CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper, which acceptance_check.py compares with the paper
PORT = "agent_as_tool_paper"
# the paper-era Plan-Execute of CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper, played and checked the same way
PE_PORT = "plan_execute_paper"
PORTS = (PORT, PE_PORT)  # the agents of the reproduction
RUN_FILE, EPISODE_FILE, CALLS_FILE, SCORES_FILE = "run.json", "episode.json", "llm_calls.json", "scores.csv"


def episode_name(agent, scenario_id, repeat):
    return f"{agent}_s{int(scenario_id):03d}_r{int(repeat)}"


def episode_dirs(run_dir):
    """The episode folders of a run in name order. A folder is an episode when it holds episode.json."""
    return sorted(p for p in Path(run_dir).iterdir() if p.is_dir() and (p / EPISODE_FILE).exists())
