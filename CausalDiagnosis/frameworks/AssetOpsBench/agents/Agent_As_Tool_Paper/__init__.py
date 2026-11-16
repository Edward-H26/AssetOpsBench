"""The paper-era Agent-As-Tool of AssetOpsBench: IBM's ReAct supervisor over one ReAct sub-agent per MCP
server, on a vendored subset of IBM's ReActXen, with raw text completions served by SGLang."""
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.supervisor_runner import MetaAgentResult, MetaAgentRunner, run_settings

__all__ = ["MetaAgentResult", "MetaAgentRunner", "run_settings"]
