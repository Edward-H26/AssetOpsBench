"""The fields the driver writes into a state file outside any call (update_state) are the ones the replay copies from the state
file of the episode as the driver wrote them (run_episodes.DRIVER_STATE_FIELDS), since no transcript line holds them; a field
added to a call of update_state without the tuple would make every replay differ."""
import ast
from pathlib import Path

from CausalDiagnosis.experiments.episode_runners import run_episodes


def written_fields(source):
    """Every field name a call of update_state in the source passes, by keyword or inside a dict literal unpacked into it."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "update_state":
            for keyword in node.keywords:
                if keyword.arg is not None:
                    names.add(keyword.arg)
                else:
                    names.update(key.value for sub in ast.walk(keyword.value) if isinstance(sub, ast.Dict)
                                 for key in sub.keys if isinstance(key, ast.Constant))
    return names


def test_every_field_update_state_writes_is_one_the_replay_takes():
    source = Path(run_episodes.__file__).read_text(encoding="utf-8")
    assert written_fields(source) == set(run_episodes.DRIVER_STATE_FIELDS)
