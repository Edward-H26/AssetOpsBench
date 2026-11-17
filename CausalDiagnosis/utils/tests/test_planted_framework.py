"""The completeness of the frameworks registry (frameworks/README.md): a second framework, planted beside AssetOpsBench with no core
file changed, is resolved by every core reader of a declaration. The plant runs in a fresh interpreter whose registry reads a
folder that holds AssetOpsBench and the planted framework (planted_framework/Planted_Check), since the core reads the
declarations when its modules are imported. Playing, judging, replaying, ledgering and showing an episode of a planted surface is
not tested here: that path of a second framework is not yet proven by a test."""
import json
import os
import subprocess
import sys
from pathlib import Path

from CausalDiagnosis.utils import frameworks
from CausalDiagnosis.utils.paths import ROOT

HERE = Path(__file__).resolve().parent
PROBE = r'''
import json, sys
from pathlib import Path
from CausalDiagnosis.utils import frameworks
frameworks.FRAMEWORKS = Path(sys.argv[1])
from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR
from CausalDiagnosis.experiments.episode_runners import run_episodes, servers_of_a_run
from CausalDiagnosis.experiments.run_verification import run_identity_file_lists as lists
from CausalDiagnosis.experiments.run_verification import invariant_digests, run_ledger
from CausalDiagnosis.utils.tests.planted_framework import planted
parser = run_episodes.build_parser()
choices = {action.dest: list(action.choices or ()) for action in parser._actions}
try:
    run_episodes.guard_test_set("planted_model", [(1, 1)], {(1, 1)}, ["a run that is not registered"])
    guard = "passed"
except SystemExit as refusal:
    guard = str(refusal)
raw = json.loads((ASSETS_DIR / "hyd_1" / "asset.json").read_text(encoding="utf-8"))
root = "CausalDiagnosis/frameworks/Planted_Check/"
print(json.dumps({
    "frameworks": list(frameworks.frameworks()),
    "agents": [a for a in run_episodes.AGENTS if a.startswith("planted")],
    "model_agents": [a for a in run_episodes.MODEL_AGENTS if a.startswith("planted")],
    "guard": guard,
    "servers": choices["servers"], "judge": choices["judge"],
    "surface_of": servers_of_a_run.surface_of("planted_iot"),
    "agent_file": list(lists.lists_of(root + "agents/runner.py")),
    "other_file": list(lists.lists_of(root + "tool_servers/server.py")),
    "roles": [lists.role_of("agent:planted_model", "planted_model"), lists.role_of("agent:planted_model", "causal"),
              lists.role_of(lists.FRAMEWORK_LIST, "causal")],
    "run_kind": run_ledger.run_kind({"kind": "planted_run"}).name,
    "profile": frameworks.surface_profile_problems(raw, "planted"),
    "families": sorted(next(f for f in invariant_digests.digest_families() if f is planted.FAMILY).scripted(None)),
    "commands": frameworks.commands().get("planted_check"),
}))
'''


def _mounted(tmp_path):
    folder = tmp_path / "frameworks"
    folder.mkdir()
    for name in frameworks.frameworks():
        (folder / name).symlink_to(frameworks.framework_dir(name), target_is_directory=True)
    (folder / "Planted_Check").symlink_to(HERE / "planted_framework" / "Planted_Check", target_is_directory=True)
    return folder


def test_a_planted_framework_is_resolved_by_every_core_reader_with_no_core_edit(tmp_path):
    run = subprocess.run([sys.executable, "-B", "-c", PROBE, str(_mounted(tmp_path))], cwd=ROOT, capture_output=True, text=True,
                         check=False, env={**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"})
    assert run.returncode == 0, run.stderr[-2000:]
    got = json.loads(run.stdout.strip().splitlines()[-1])
    assert got["frameworks"] == sorted([*frameworks.frameworks(), "Planted_Check"])
    assert set(got["agents"]) == {"planted_check", "planted_model"}
    assert got["model_agents"] == ["planted_model"]  # an agent without the flag model_agent counts as a model agent
    assert "--agent planted_model would play the test combination" in got["guard"]
    assert "planted" in got["servers"] and "planted_judge" in got["judge"] and got["surface_of"] == "planted"
    assert got["agent_file"] == ["agent:planted_model"] and got["other_file"] == ["framework"]
    assert got["roles"] == ["identity", "bound", "bound"]
    assert got["run_kind"] == "planted_run"
    assert got["profile"] == ["the planted surface's profile check ran"]
    assert got["families"] == ["scripted.planted"]
    assert got["commands"] == {"planted-command": "CausalDiagnosis.utils.tests.planted_framework.planted"}
