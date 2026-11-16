"""The episode functions of the AssetOpsBench rows: the rig's state file functions bound to the engine
of the rows and to their briefing. The replay rebuilds RecordedRigEnv from a state file, the writer of the released
evidence puts the briefing of the rows into task_text, which the driver hands the agent as its question and the judge
reads, and the start creates an episode with both.
"""
from functools import partial

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import briefing
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

replay = partial(episode.replay, env_class=RecordedRigEnv)
write_released = partial(episode.write_released, compose=briefing)
start = partial(episode.start, env_class=RecordedRigEnv, compose=briefing)
