"""The fixtures and hooks of the drivers' tests, shared with this folder, and the generated tables of the scripted runs."""
import pytest

from CausalDiagnosis.experiments.episode_runners.tests.conftest import *  # noqa: F401,F403
from CausalDiagnosis.experiments.episode_runners.tests.conftest import pytest_configure  # noqa: F401
from CausalDiagnosis.experiments.episode_runners.tests.conftest import own_table_copies


@pytest.fixture(scope="session")
def replay_tables(tmp_path_factory):
    """The folder of the generated tables that the scripted runs (test_replay_check.py, scripted) and their replays read and write,
    copies of the session copies taken once, so that a test that adds the table of a row to a session copy changes no table that a
    scripted run recorded. It is defined here, since a test module that imports a session fixture builds an instance of its own,
    and every module that takes scripted reads the one folder that the runs were made on."""
    folder = tmp_path_factory.mktemp("replay_tables")
    with pytest.MonkeyPatch.context() as patch:
        own_table_copies(patch, folder)
    return folder
