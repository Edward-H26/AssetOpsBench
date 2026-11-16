import asyncio
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import RIG_SERVER  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier  # noqa: E402
from CausalDiagnosis.environment.reference_policies.tier_classifiers import TierClassifiers  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv  # noqa: E402
from CausalDiagnosis.environment.reference_policies.greedy_policy import GreedyPolicy, ServedClassifiers  # noqa: E402
from CausalDiagnosis.environment.reference_policies.policies import DeclarePolicy, DoNothingPolicy, OraclePolicy, RandomPolicy, run_episode  # noqa: E402
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode  # noqa: E402
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.policy_through_tools import play, tool_call  # noqa: E402
from CausalDiagnosis.agents.utils.mcp_sessions import open_sessions  # noqa: E402


# ----- reference policies through the server -----
def make_policy(name, env, classifiers, seed):
    if name == "oracle":
        return OraclePolicy(env)
    if name == "do_nothing":
        return DoNothingPolicy()
    if name == "greedy":
        return GreedyPolicy(classifiers)
    if name == "random":
        return RandomPolicy(seed)
    if name == "declare":
        return DeclarePolicy()
    raise ValueError(f"unknown policy {name}")


def play_policy(name, state_path, server_path, cycles, blocks, config, seed, classifiers=None):
    """One episode of a reference policy through the action server, played by policy_client.play on the tool answers. A
    shadow environment in this process is reset with the same seed and configuration as the server's; the oracle reads
    the hidden state from it, and after every call the shadow steps the same action with the think cycles the server
    recorded, so the two environments stay identical. Returns whether the served log equals the shadow's."""
    shadow = HydraulicRigEnv(cycles, blocks, config)
    shadow.reset(seed)
    policy = make_policy(name, shadow, classifiers, seed)

    async def played():
        # one set of sessions for the episode, as every runner has: policy_client.play runs in a worker thread and sends each
        # action to the session on this loop
        async with open_sessions({RIG_SERVER: server_path}) as sessions:
            loop = asyncio.get_running_loop()

            def send(action):
                tool, arguments = tool_call(action)
                text, _is_error = asyncio.run_coroutine_threadsafe(sessions.call(RIG_SERVER, tool, arguments), loop).result()
                state = episode.load_state(state_path)
                if not shadow.done:
                    shadow.step(action, think_cycles=state["think_cycles"][-1] if state["think_cycles"] else 0)
                return text

            await asyncio.to_thread(play, policy, episode.opening(episode.load_state(state_path)), send)

    asyncio.run(played())
    served = episode.replay(episode.load_state(state_path))
    return served.log == shadow.log


def oracle_reference(cycles, blocks, config, seed):
    """The oracle policy's episode on the same seed and configuration: the verifier's own cost of it, which normalizes
    the agent's cost."""
    env = HydraulicRigEnv(cycles, blocks, config)
    log = run_episode(env, OraclePolicy(env), seed)
    return verifier.verify(log, verifier.released_from_env(env), env.oracle(), None)["cost"]


_SERVED_CLASSIFIERS = {}  # by the identity of the recorded cycles, which the entry keeps alive so that the identity stays theirs


def served_classifiers(cycles, blocks):
    """The observation model of the greedy policy, fitted once per process."""
    if id(cycles) not in _SERVED_CLASSIFIERS:
        _SERVED_CLASSIFIERS[id(cycles)] = (cycles, ServedClassifiers(TierClassifiers(cycles, blocks), cycles))
    return _SERVED_CLASSIFIERS[id(cycles)][1]


def greedy_reference(cycles, blocks, config, seed):
    """The sensor reads of the greedy policy's episode on the same seed and configuration, the ceiling per sensor group that
    the verifier scope "all" holds the agent's reads against, since this policy has to read to diagnose."""
    env = HydraulicRigEnv(cycles, blocks, config)
    return verifier.read_counts(run_episode(env, GreedyPolicy(served_classifiers(cycles, blocks)), seed))
