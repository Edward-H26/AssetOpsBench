from pathlib import Path


HERE = Path(__file__).resolve().parent
from CausalDiagnosis.utils import frameworks  # noqa: E402


ROUTER_BASE_URL = "https://api.tokenrouter.com/v1"
DEFAULT_MODEL = "MiniMax-M3"
# the agent kinds the frameworks declare, read as data (utils/frameworks.py): a declared agent without the flag model_agent counts
# as a model agent, so the serving check holds for it
DECLARED_AGENTS = frameworks.agents()
MODEL_AGENTS = (*(name for name, entry in DECLARED_AGENTS.items() if entry.get("model_agent", True)), "static", "causal")
# the declared agents that reach the model on the completion route themselves, each played by the entry of its declaration
PAPER_ERA_AGENTS = tuple(name for name, entry in DECLARED_AGENTS.items() if entry.get("route") == "completion")
POLICY_AGENTS = ("oracle", "do_nothing", "greedy", "random", "declare")
# the declared checks without a model, each with a backend of its own (the scripted wiring check)
SCRIPTED_AGENTS = tuple(name for name in DECLARED_AGENTS if name not in MODEL_AGENTS)
AGENTS = (*SCRIPTED_AGENTS, *MODEL_AGENTS, *POLICY_AGENTS)
ABSTENTION_RULES = ("decision", "fixed_cut")


# the modules through which the driver reaches the code of each agent label beyond what it imports for every run, which the
# test of the agent files imports alone in a fresh interpreter; static and the reference policies run no agent code
AGENT_ENTRY_MODULES = {**{name: entry["modules"] for name, entry in DECLARED_AGENTS.items() if entry.get("modules")},
                       "causal": ("CausalDiagnosis.agents.Causal_Diagnosis_Agent.controller", "CausalDiagnosis.agents.Causal_Diagnosis_Agent.tool_surfaces")}


def agent_label_of(agent, model, causal_options=None):
    """The agent column of scores.csv: a policy or the scripted plan by its name, a model agent with its model, and the causal
    agent with one suffix per ablation switch, so that the tables separate the ablation rows."""
    if agent in POLICY_AGENTS or agent in SCRIPTED_AGENTS:
        return agent
    if agent == "causal":
        # here and not at the top, so that a run of another agent or of a reference policy loads no file of the causal agent
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.agent_options import CausalLoopOptions

        label = CausalLoopOptions(**(causal_options or {})).label()
        return f"{label}_{model}"
    return f"{agent}_{model}"
