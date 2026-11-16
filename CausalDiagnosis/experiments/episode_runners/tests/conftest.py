"""The drivers import the fork's src and their own folder by path; the tests run from the fork root with the same paths.
The models of the diagnostics server are generated, not committed, so they are fitted once before the tests that need
them; when the server package cannot be imported (its modules are being rebuilt) the fit is skipped and only the tests
that start a diagnostics server fail."""
import json
import sys
from pathlib import Path

DRIVERS = Path(__file__).resolve().parents[1]
from CausalDiagnosis.utils.paths import SRC
for path in (str(SRC),):
    if path not in sys.path:
        sys.path.insert(0, path)


import pytest  # noqa: E402

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "real_start_check: the test runs the start check and the guard of the test set of run_episodes.py as they are")
    config.addinivalue_line("markers", "registered_file: the test reads registered_analysis.json as it is and not the copy of the tests")
    config.addinivalue_line("markers", "offline_file: the test reads the offline table of the MCP in the results folder as it is and not "
                                       "the copy of the tests")


@pytest.fixture(autouse=True)
def unguarded_test_set(request, monkeypatch):
    """The one seam of the start check, in test code alone: the tests play model agents against stub endpoints on test
    combinations, so the guard of a test holds no test combination. The start check itself runs as it is, so a run of a test is
    stamped as every run of a tree that is no clean repository on origin, and the tables read it as such. The tests that prove the check and the
    guard carry the mark real_start_check and meet the functions of run_episodes/ as they are."""
    if request.node.get_closest_marker("real_start_check"):
        yield
        return
    from CausalDiagnosis.experiments.episode_runners import run_episodes

    monkeypatch.setattr(run_episodes.start_check, "guarded_starts", lambda *args, **kwargs: set())
    yield


@pytest.fixture(scope="session")
def registered_test_copy(tmp_path_factory):
    """The copy of registered_analysis.json that the tests read, written once for the session: the smallest number of clusters of
    an interval and the largest unscored share of a row open."""
    from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered

    data = registered.load()
    data["intervals"]["min_clusters"] = None
    data["counting"]["max_unscored_share"] = None
    path = tmp_path_factory.mktemp("registered") / registered.PATH.name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def read_the_registered_test_copy(request, monkeypatch):
    """The tests read registered_test_copy. A root of a few starts would print every interval blank under the registered smallest
    number of clusters, and the registered largest unscored share would refuse every row with one unscored episode; the tests
    were written with no such limit, so no assertion changes. The effect of each number is
    tested on a copy of its own, and the tests that read the file as it is carry the mark registered_file."""
    if request.node.get_closest_marker("registered_file"):
        yield
        return
    from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered

    monkeypatch.setattr(registered, "PATH", request.getfixturevalue("registered_test_copy"))
    yield


@pytest.fixture(autouse=True)
def read_no_offline_table_of_record(request, tmp_path_factory, monkeypatch):
    """paper_tables/ reads the offline table of the MCP in the results folder (main_tables.OFFLINE_PATH), which the cluster writes
    after the grid. The tests read an absent copy, so that no page of a test changes with the table of record; a test that writes an
    offline table points OFFLINE_PATH at it, and the tests that read the file as it is carry the mark offline_file."""
    if request.node.get_closest_marker("offline_file"):
        yield
        return
    from CausalDiagnosis.evaluation.result_tables import paper_tables as main_tables

    monkeypatch.setattr(main_tables.read_sources, "OFFLINE_PATH", tmp_path_factory.getbasetemp() / "no_offline_table" / main_tables.OFFLINE)
    yield


@pytest.fixture(scope="session", autouse=True)
def fitted_models():
    if not Path(default_facts_path()).exists():
        return
    try:
        from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import ensure_models
    except ImportError:
        return
    ensure_models()


@pytest.fixture
def registered_root_numbers(tmp_path, monkeypatch):
    """The test copy of the registered file with the numbers that the tables of a registered root need (registered.TABLE_VALUES) set
    where they limit nothing, every interval printed and no row refused, for the tests of registered roots."""
    from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered

    data = json.loads(registered.PATH.read_text(encoding="utf-8"))
    data["intervals"]["min_clusters"] = 1
    data["counting"]["max_unscored_share"] = 1.0
    path = tmp_path / "registered_root" / registered.PATH.name
    path.parent.mkdir()
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(registered, "PATH", path)
    return path
