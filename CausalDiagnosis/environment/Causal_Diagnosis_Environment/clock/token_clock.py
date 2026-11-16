"""The token clock of the clock mode generated_tokens: the thinking seconds of the tokens an agent's model generated between
two calls (token_seconds, seconds_since_last_call), which become cycles by the rule of the wall clock (wall_clock.py), and
the ledger of those tokens in one episode, which the clock mode generated_tokens charges as the agent's thinking time (CausalDiagnosis/docs/environment.md, section Time). The driver opens the ledger empty in the episode
folder, the folder of the state file and of the transcript, before the agent starts an episode of that mode, and names that
folder for the time of the episode (episode); the harness of the agent adds the generated tokens of each model call before
the reply goes back to the agent (add_generated); the servers read the total at every tool call. An episode of another mode
has no ledger, and an addition to a folder without one does nothing, so that its folder keeps its files. No import of the MCP
package, so that the harness and the servers import this module alike."""
import json
from contextlib import contextmanager
from pathlib import Path

from CausalDiagnosis.utils.atomic_writes_and_locks import locked

NAME = "generated_tokens.jsonl"
# the episode folder that the driver names for the time of one episode, whose ledger add_generated writes to, None outside an
# episode. One process plays one episode at a time, and the runner of an episode returns only after every model call of the
# episode has, so the folder of a call is the folder of its episode.
_FOLDER = None


def open_ledger(folder):
    """Start the empty ledger of an episode in its folder, replacing one that a played episode left."""
    path = Path(folder) / NAME
    with locked(path):
        path.write_text("", encoding="utf-8")
    return path


def add(folder, tokens):
    """Add the generated tokens of one model call to the ledger of an episode folder, under the lock of the file, and nothing
    when the folder holds no ledger."""
    path = Path(folder) / NAME
    if not path.is_file():
        return
    with locked(path):
        with path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps({"tokens": int(tokens)}) + "\n")


def total(folder):
    """The generated tokens of the ledger of an episode folder in all, None when the folder holds no ledger."""
    path = Path(folder) / NAME
    if not path.is_file():
        return None
    with locked(path):
        lines = path.read_text(encoding="utf-8").splitlines()
    return sum(json.loads(line)["tokens"] for line in lines if line)


@contextmanager
def episode(folder):
    """Name the folder of the episode the driver plays, for the time of the episode, so that the harness of any agent adds to
    its ledger without knowing the folder."""
    global _FOLDER
    before, _FOLDER = _FOLDER, folder
    try:
        yield
    finally:
        _FOLDER = before


def episode_folder():
    """The folder of the episode the driver plays, None outside an episode; the ledger of failed model attempts
    (CausalDiagnosis.utils.model_failures) is written to it too."""
    return _FOLDER


def add_generated(tokens):
    """Add the generated tokens of one model call to the ledger of the episode the driver plays: nothing outside an episode,
    for a call whose server reported no count, or in an episode whose folder holds no ledger, the episode of another mode."""
    if _FOLDER is not None and tokens is not None:
        add(_FOLDER, tokens)


def token_seconds(config, tokens):
    """The thinking seconds of the tokens an agent's model generated between two calls, in the generated_tokens mode."""
    return tokens * config.seconds_per_token


def seconds_since_last_call(state, config, generated):
    """The thinking seconds of a call in the generated_tokens mode: the tokens generated since the previous call, the total
    of the ledger generated less the total read before, kept in the state as think_tokens, with generated as the new
    generated_tokens_read, times seconds_per_token."""
    tokens = generated - state.get("generated_tokens_read", 0)
    state.setdefault("think_tokens", []).append(tokens)
    state["generated_tokens_read"] = generated
    return token_seconds(config, tokens)
