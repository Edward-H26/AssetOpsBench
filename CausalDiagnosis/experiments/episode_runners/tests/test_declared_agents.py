"""The agent kinds of the driver come from the declarations of the frameworks (utils/frameworks.py): a declared agent without the
flag model_agent counts as a model agent, so the guard of the test set refuses it on a start of the test set in a run that is not
registered, and the ports of AssetOpsBench are played through the entries of their declaration."""
import subprocess
import sys

from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.utils.paths import PACKAGE, ROOT, SRC

PLANTED = 'AGENTS = {"planted": {"route": "chat", "surfaces": ("aob",)}}\n'


def test_a_declared_agent_without_the_model_flag_counts_as_a_model_agent_and_is_refused_on_a_test_start(tmp_path):
    for name, text in (("AssetOpsBench", (PACKAGE / "frameworks" / "AssetOpsBench" / "__init__.py").read_text(encoding="utf-8")),
                       ("Planted", PLANTED)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text(text, encoding="utf-8")
    code = (f"import sys\nsys.path[:0] = [{str(ROOT)!r}, {str(SRC)!r}]\nfrom pathlib import Path\n"
            f"from CausalDiagnosis.utils import frameworks\nframeworks.FRAMEWORKS = Path({str(tmp_path)!r})\n"
            "from CausalDiagnosis.experiments.episode_runners import run_episodes as r\n"
            "assert 'planted' in r.MODEL_AGENTS and 'planted' in r.AGENTS and 'planted' not in r.PAPER_ERA_AGENTS\n"
            "try:\n    r.guard_test_set('planted', [(1, 2)], [(1, 2)], ['not registered'])\nexcept SystemExit as e:\n    print(e)\n")
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, cwd=ROOT, check=True).stdout
    assert out.startswith("--agent planted would play the test combination [1, 2]")


def test_the_ports_of_assetopsbench_are_model_agents_played_through_their_entries():
    assert run_episodes.PAPER_ERA_AGENTS == ("plan_execute", "agent_as_tool")
    assert set(run_episodes.PAPER_ERA_AGENTS) < set(run_episodes.MODEL_AGENTS)
    assert run_episodes.AGENTS[0] == "scripted" and "scripted" not in run_episodes.MODEL_AGENTS
