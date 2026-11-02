"""OpenEnv interface of the hydraulic rig gym: reset, step and state with typed action, observation and state
models after the core of External/meta-pytorch_OpenEnv (docs/hydraulic_rig/environment.md, section Gym and OpenEnv).

Simulation mode, this module: every agent facing tool is an action that goes through step, so that it counts as a
step, advances the clock, lands in the log and ends the episode when the environment says so, and the terminal
observation carries the verifier's reward. Production mode: the same actions are the MCP tools of the action
server servers.hydraulic_rig (RFC 003 of OpenEnv, every action is a tool), served over MCP to a client that brings
its own agent, where no reward, step count or done flag exists. The tool names are the action types of the
environment in both modes. The action server advertises flat typed parameters for the Plan-Execute argument
filler, so there submit_diagnosis takes the four levels and declare_insufficient_evidence the separating test as
flat parameters, while the OpenEnv action carries the environment's own argument dict
(docs/hydraulic_rig/action-server.md).

openenv is optional and imported lazily; the package is openenv.core, of which openenv_core is the deprecated
alias. With it installed the models subclass the core's Action, Observation and State and the environment
subclasses the core's Environment, so that the core's create_app can serve it; without it, pydantic models with
the same fields and a plain base class take their place. The reward of the terminal observation is None when the
verifier could not run, which OpenEnv keeps apart from a score of zero; the Gymnasium wrapper underneath reports
the same case as 0.0 with truncated set and info["reward"] None, since gymnasium needs a number in its tuple.
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from .config import EpisodeConfig
from .data import default_facts_path
from .gym_env import HydraulicRigGym

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State

    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False

    class Environment:
        """Stand in for the core's Environment when openenv is not installed."""

        SUPPORTS_CONCURRENT_SESSIONS = False

    class Action(BaseModel):
        """The core's Action: unknown fields are rejected."""

        model_config = ConfigDict(extra="forbid")
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        """The core's Observation: the done flag, the reward and free metadata."""

        model_config = ConfigDict(extra="forbid")
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        """The core's State: episode bookkeeping, open to more fields."""

        model_config = ConfigDict(extra="allow")
        episode_id: Optional[str] = None
        step_count: int = 0


class HydraulicRigAction(Action):
    """One tool call in the shape of the core's CallToolAction: the tool name is the action type of the
    environment and the arguments are the action's arguments."""

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class HydraulicRigObservation(Observation):
    """What the policy reads: the briefing and the asset profile at reset, the answer envelope after a step."""

    text: str


class HydraulicRigState(State):
    """Episode bookkeeping for the orchestrator. The verdict is the verifier's and is set when the episode ends;
    it holds the hidden truth, so the state belongs to the infrastructure boundary and never reaches the agent."""

    seed: Optional[int] = None
    time: int = 0
    done: bool = False
    end_reason: Optional[str] = None
    verdict: Optional[Any] = None


class HydraulicRigOpenEnv(Environment):
    """The hydraulic rig gym behind the OpenEnv reset, step and state interface.

    The arguments default to EpisodeConfig() and default_facts_path() because the OpenEnv server builds one
    instance per session by calling the class without arguments. Every instance owns its environment and its
    oracle copy; the recorded cycles are shared read only.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, config=None, facts_path=None, seed_offset=0):
        super().__init__()
        self.gym = HydraulicRigGym(config or EpisodeConfig(), facts_path or default_facts_path(), seed_offset)
        self._state = HydraulicRigState()

    def reset(self, seed=None, episode_id=None, **kwargs):
        text, info = self.gym.reset(seed)
        # the default episode id names the seed, so that a replay of the same seed reports the same state
        self._state = HydraulicRigState(
            episode_id=episode_id or f"hyd-{info['seed']}", seed=info["seed"], time=self.gym.env.time
        )
        return HydraulicRigObservation(text=text)

    def step(self, action, timeout_s=None, **kwargs):
        if isinstance(action, dict):
            action = HydraulicRigAction.model_validate(action)
        text, _reward, terminated, truncated, info = self.gym.step(dict(action.arguments, type=action.tool_name))
        done = terminated or truncated
        self._state.step_count += 1
        self._state.time = info["time"]
        reward = 0.0
        if done:
            self._state.done = True
            self._state.end_reason = info["end_reason"]
            self._state.verdict = info["verdict"]
            reward = info["reward"]
        return HydraulicRigObservation(
            text=text, done=done, reward=reward, metadata={"cost": info["cost"], "time": info["time"]}
        )

    @property
    def state(self):
        return self._state
