"""One lock rule for a call on an episode's state file: the action server's own call path and the
metering layer's sessions both wait for CausalDiagnosis.utils.atomic_writes_and_locks.locked on the state file, so the calls of several server processes
on one episode run one at a time."""
import threading
import time

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import StateFileSessions
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
from CausalDiagnosis.utils.atomic_writes_and_locks import locked


def held_while(state, target):
    """Run target in a thread while the state file's lock is held; return whether it was still waiting after 0.3 s."""
    caller = threading.Thread(target=target)
    with locked(state):
        caller.start()
        time.sleep(0.3)
        waiting = caller.is_alive()
    caller.join(5)
    return waiting, caller.is_alive()


def test_the_action_server_call_waits_for_the_lock(tmp_path):
    state = tmp_path / "episode.json"
    outcome = []

    def call():
        try:
            episode.call(state, "finish", {}, {"type": "finish"})
        except OSError as error:  # no state file: the call reached load_state only after the lock was released
            outcome.append(type(error).__name__)

    waiting, still_alive = held_while(state, call)
    assert waiting and not still_alive and outcome == ["FileNotFoundError"]


def test_the_metering_session_waits_for_the_same_lock(tmp_path):
    state = tmp_path / "episode.json"
    outcome = []

    def enter():
        with pytest.raises(OSError):
            with StateFileSessions(state)(None):  # the lock is taken before the entry is read
                pass
        outcome.append("released")

    waiting, still_alive = held_while(state, enter)
    assert waiting and not still_alive and outcome == ["released"]
