"""AssetOpsBench's runner (src/agent/plan_execute/executor.py) turns the path of a server into its start in _make_stdio_params. A
launcher that the driver writes beside the files of an episode starts isolated, as python -I <launcher path>, which its first
lines require, whether its run folder lies inside the repository, as under the default --out, or outside it, and whatever dots
the names of its folders hold, so no --out is refused for a dot; a module of a
package of the repository starts as upstream starts it, as python -m <dotted path> from the repository root. The error output of
a server follows the same rule (_server_errlog): a launcher writes it to hidden/ of its episode folder, a module of a package to
the caller's standard error."""
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.agents import upstream_agent_package  # noqa: F401  registers upstream's agent package first
from agent.plan_execute import executor
from agent.runner import DEFAULT_SERVER_PATHS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET
from CausalDiagnosis.experiments.episode_runners.episode_launchers import write_launchers
from CausalDiagnosis.experiments.episode_runners.run_episodes.run_record import default_out, run_base, run_label
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_name
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import RIG_SERVER


def _start(server):
    """(command, arguments, working directory) of the start of a server path."""
    params = executor._make_stdio_params(server)
    return params.command, params.args, params.cwd


def _episode_launchers(folder):
    """The launcher files of an episode of the scripted check, as the driver writes them into a new episode folder."""
    folder.mkdir(parents=True)
    return list(write_launchers(folder).values())


@pytest.mark.parametrize("out", [default_out(DEFAULT_ASSET), "CausalDiagnosis/agents"])
def test_a_launcher_in_a_run_folder_inside_the_repository_starts_isolated(tmp_path, out):
    """The launchers of an episode of a run under the default --out and of a run under an --out below a package folder of the
    tree. The driver creates the run folder and the episode folder and writes no __init__.py into either, so the launchers are
    written in pytest's temporary folder and only their names are placed under the repository root."""
    episode = executor._REPO_ROOT / out / run_label("static", "scripted") / episode_name(0)
    for written in _episode_launchers(tmp_path / "written"):
        launcher = episode / written.name
        assert _start(launcher) == ("python", ["-I", str(launcher)], None)


def test_a_run_folder_under_an_out_folder_with_a_dot_inside_the_repository_is_not_refused(tmp_path):
    """run_base gives the folder of a run under an --out inside the repository whose folder name holds a dot, and the launchers of
    an episode there start isolated as anywhere else. Only the names are placed under the repository root."""
    out = default_out(DEFAULT_ASSET) + "/with.dot"
    run_dir = run_base(out, run_label("static", "scripted"))
    assert run_dir == (executor._REPO_ROOT / out / run_label("static", "scripted")).resolve()
    for written in _episode_launchers(tmp_path / "written"):
        launcher = run_dir / episode_name(0) / written.name
        assert _start(launcher) == ("python", ["-I", str(launcher)], None)


def test_a_launcher_in_a_run_folder_outside_the_repository_starts_isolated(tmp_path):
    for launcher in _episode_launchers(tmp_path / run_label("static", "scripted") / episode_name(0)):
        assert not launcher.is_relative_to(executor._REPO_ROOT)
        assert _start(launcher) == ("python", ["-I", str(launcher)], None)


@pytest.mark.parametrize("name", sorted(DEFAULT_SERVER_PATHS))
def test_a_server_module_of_the_repository_starts_as_a_module_from_the_repository_root(name):
    module = executor._REPO_ROOT / "src" / "servers" / name / "main.py"
    assert module.is_file()
    assert _start(module) == ("python", ["-m", f"src.servers.{name}.main"], str(executor._REPO_ROOT))


def test_the_error_output_of_a_server_follows_the_same_rule(tmp_path, monkeypatch):
    """Under a repository root in pytest's temporary folder, a module of a package writes its error output to the caller's
    standard error and leaves its package folder as it is, and a launcher in an episode folder writes it to hidden/ there."""
    monkeypatch.setattr(executor, "_REPO_ROOT", tmp_path)
    package = tmp_path / "servers"
    package.mkdir()
    for name in ("__init__.py", "main.py"):
        (package / name).write_text("", encoding="utf-8")
    with executor._server_errlog(package / "main.py", "servers") as log:
        assert log is sys.stderr
    assert sorted(path.name for path in package.iterdir()) == ["__init__.py", "main.py"]
    episode = tmp_path / "runs" / episode_name(0)
    episode.mkdir(parents=True)
    launcher = write_launchers(episode, servers=(RIG_SERVER,))[RIG_SERVER]
    with executor._server_errlog(launcher, RIG_SERVER) as log:
        assert Path(log.name) == launcher.parent / "hidden" / f"{RIG_SERVER}.0.stderr"
