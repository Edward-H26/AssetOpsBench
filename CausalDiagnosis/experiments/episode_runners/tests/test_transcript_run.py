"""The transcript of an episode as a run of the driver leaves it (CausalDiagnosis/mcp_servers/utils/transcript.py): one file per episode, written
by the server processes of the episode, ordered across them, with the hash of every answer, and the hashes of the tool
listings in run.json. The tests assert relations between the files of a scripted run, never a recorded value. Run folders go
under pytest's tmp_path, outside the repository."""
import importlib.util
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.mcp_servers.utils import transcript  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.tests.test_run_episodes import _run, _values  # noqa: E402

# the rig's action server on the static preset, and the AssetOpsBench servers behind the metering layer on the base row
RUNS = {"rig": ["--config", "static"], "aob": ["--config", "base", "--servers", "aob"]}


@pytest.mark.parametrize("surface", sorted(RUNS))
def test_a_scripted_episode_writes_one_transcript_ordered_across_its_servers(tmp_path, surface):
    run_dir, rows, meta = _run([*RUNS[surface], "--agent", "scripted", "--seeds", "1"], tmp_path / surface)
    assert len(rows) == 1, rows  # one episode, one row of scores.csv
    assert rows[0]["error"] == "", rows[0]["error"]
    ep_dir = run_dir / "seed000000"
    lines = transcript.read(ep_dir)  # the reader refuses a torn or renumbered file
    assert lines[0]["kind"] == "start" and {line["kind"] for line in lines[1:]} == {"handover", "ready", "list", "enter", "call", "exit"}
    calls = [line for line in lines if line["kind"] == "call"]
    steps = [step for step in read_json(ep_dir / "trajectory.json") if step["tool"]]
    # the servers wrote one line per call that the runner recorded, in the runner's order, and hashed the text it received
    assert [(call["server"], call["tool"]) for call in calls] == [(step["server"], step["tool"]) for step in steps]
    assert [call["answer_sha256"] for call in calls] == [transcript.text_sha256(step["response"]) for step in steps]
    # a server that changes the episode stamps its line with the wall clock reading of the call, in rising order
    walls = [call["wall"] for call in calls if "wall" in call]
    state = read_json(ep_dir / "episode.json")
    assert walls == sorted(walls) and len(walls) == len(state["tool_calls"]) and lines[0]["wall"] == state["episode_started_at"]
    assert [call["wall"] for call in calls if call["stepped"]] == state["called_at"]
    # only the diagnostics server keeps the text of its answers, which no other file of a server holds
    kept = [call for call in calls if "answer" in call]
    assert kept and {call["server"] for call in kept} == {"causal_diagnosis"}
    assert [call["answer"] for call in kept] == [step["response"] for step in steps if step["server"] == "causal_diagnosis"]
    assert all(call["released_sha256"] for call in calls)
    # run.json names the hash of the listing that every registered server sent, and every later listing of a server equals it
    assert set(meta["tool_listings"]) == set(meta["servers"])
    listings = [line for line in lines if line["kind"] == "list"]
    assert {line["server"] for line in listings} == set(meta["servers"])
    assert all(line["listing_sha256"] == meta["tool_listings"][line["server"]] for line in listings)
    # no line holds a draw of the hidden samplers, and the ledger of the run hashes the transcript with the other files
    reset = read_json(ep_dir / HIDDEN / "log.json")[0]
    draws = [reset["hidden"], *([reset["schedule"]] if reset["schedule"] else []), *reset["schedule"]]
    assert not any(value in draws for line in lines for value in _values(line))
    hashed = read_json(run_dir / "ledger.json")["episodes"]["seed000000"]["files"]
    assert hashed[transcript.NAME]["sha256"] == transcript.file_sha256(ep_dir / transcript.NAME)
