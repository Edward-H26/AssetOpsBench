"""The blocks that the fits of the belief leave out, through the driver: the scripted wiring check, with a ranking
after its reading that follows the repair, plays one development start of four faults, none at its worst level, on the base row,
where its repair of the pump changes the combination, on the deterioration row with its drop card on a fixed schedule and the
declared clock, where the pump repair does not take, the first drop is never read and the second is, and on the AssetOpsBench
rows. The server finds exactly the blocks of the readings the agent was served, an unread drop none, and no file of the episode
holds them; no number that names a combination moves with them while the belief does; the replay of an episode reproduces every
answer, and a server that leaves out the start's block alone does not; only the server and the yardstick look blocks up. No test
states a value computed from the rig data."""
import json
import shutil
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS
from CausalDiagnosis.frameworks.AssetOpsBench.agents import scripted_wiring_check
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
from CausalDiagnosis.experiments.run_verification import replay_check
from CausalDiagnosis.experiments.run_verification.replay_check import run_record_checks
from CausalDiagnosis.experiments.table_definitions.training_variants import asset_of
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import effect_model as em
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_clean_fits import needs_facts
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools, Models
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.utils.paths import ROOT

START = (20, 90, 1, 115)
START_STATE = "20/90/1/115@0"
PRESETS = Path(ROOT) / "CausalDiagnosis" / "configs" / "layer_presets"
EPISODE = "seed000000"
pytestmark = needs_facts


def preset(folder, row):
    """The row's preset on development starts, the drop row with its drop card on a fixed schedule and the declared clock."""
    spec = json.loads((PRESETS / ("base.json" if row == "repair" else "deterioration.json")).read_text(encoding="utf-8"))
    spec = {**spec, "name": f"played_blocks_{row}", "fault_rule": {"min_faulty": 3, "max_faulty": 4}}
    if row == "drop":
        spec["switches"] = {**spec.get("switches", {}), "clock_mode": "declared",
                            "deterioration": {"mode": "schedule", "components": 1, "first_drop": [4, 4], "interval": 3}}
    path = Path(folder) / f"{row}.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def with_ranking_after_the_change(profile, aob=False):
    """The plan of the scripted wiring check with a ranking after its reading that follows the repair, the one answer of the plan
    that reads the belief after a change of the combination."""
    steps = PLAN(profile, aob)
    after = next((i for i, step in enumerate(steps) if step[0].endswith("after the repair")), None)
    ranking = ("Rank the candidate fault combinations after the repair", CAUSAL_SERVER, "rank_root_causes", '{"top_k": 3}')
    return steps if after is None else [*steps[: after + 1], ranking, *steps[after + 1:]]


PLAN = scripted_wiring_check.scripted_steps


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scripted_wiring_check, "scripted_steps", with_ranking_after_the_change)
        for row in ("repair", "drop"):
            folder = tmp_path_factory.mktemp(row)
            out[row] = Path(run_episodes.main(["--config", str(preset(folder, row)), "--start-state", START_STATE, "--agent", "scripted",
                                               "--seeds", "1", "--no-grade", "--out", str(folder / "runs")]))
        folder = tmp_path_factory.mktemp("aob")
        out["aob"] = Path(run_episodes.main(["--config", "static", "--agent", "scripted", "--servers", "aob", "--store", "memory",
                                             "--seeds", "1", "--no-grade", "--out", str(folder / "runs")]))
    return out


def served(log):
    """[(reading number, combination)] of every reading the engine served, from the hidden log."""
    return [(int(o["n"]), tuple(o["combination"])) for e in log if e.get("event") == "read" for o in e.get("served") or ()]


@pytest.mark.parametrize("row", ("repair", "drop", "aob"))
def test_the_server_finds_the_blocks_of_the_readings_served_and_no_file_holds_them(runs, row):
    episode = runs[row] / EPISODE
    log, released = read_json(episode / "hidden" / "log.json"), read_json(episode / "released.json")
    expected = list(dict.fromkeys(h for _n, h in sorted(served(log))))
    assert load_models().served_blocks(released["observations"]) == expected
    assert not (episode / "played_blocks.json").exists()
    if row != "aob":
        assert expected[0] == START and len(expected) == 2
    if row == "drop":
        drops = [e for e in log if e.get("event") == "scheduled_drop"]
        assert len(drops) >= 2 and any(e.get("event") == "order_ineffective" for e in log)
        first = list(START)
        first[COMPONENTS.index(drops[0]["component"])] = drops[0]["to_level"]
        assert tuple(first) not in expected  # the first drop is never read


@pytest.mark.parametrize("row", ("repair", "drop"))
def test_no_number_that_names_a_combination_moves_with_the_blocks_served_and_the_belief_does(runs, row, tmp_path, monkeypatch):
    """On the final evidence of the episode, with the blocks the server finds and with the start's block alone: the anomaly scores
    of every reading and the predicted summaries of an estimate, those of the start's fits for its named combination, are the
    same numbers, while the belief, its effective number of hypotheses, is not."""
    episode, fits = runs[row] / EPISODE, load_models().without(START)
    scores, estimates, beliefs = {}, {}, {}
    for case in ("served", "start_only"):
        folder = tmp_path / case
        folder.mkdir()
        shutil.copy(episode / "released.json", folder / "released.json")
        if case == "start_only":
            monkeypatch.setattr(Models, "served_blocks", lambda self, observations: [START])
        tools = CausalTools(fits, folder / "released.json")
        scores[case] = [(item.evidence_id, item.anomaly_score) for item in tools.get_episode_evidence("all").items]
        estimate = tools.estimate_intervention_effect("replace:cooler")
        record = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))[-1]
        estimates[case] = ([s.mean for s in estimate.predicted], list(fits.effect.mu(em.parse_combo(record["predicted_combination"]))))
        beliefs[case] = tools.rank_root_causes(1).effective_hypotheses
    assert scores["served"] == scores["start_only"]
    assert all(means == expected for means, expected in estimates.values())
    assert beliefs["served"] != beliefs["start_only"]


@pytest.mark.parametrize("row", ("repair", "drop"))
def test_the_replay_reproduces_every_answer_and_a_server_that_leaves_out_the_start_alone_does_not(runs, row, monkeypatch, tmp_path):
    meta = read_json(runs[row] / "run.json")
    store, _reason = run_record_checks.store_of(meta)
    world = load_world(None, asset_of(meta))
    assert replay_check.replay_episode(runs[row] / EPISODE, meta, tmp_path / "replayed", store, world)["problems"] == []
    monkeypatch.setattr(Models, "served_blocks", lambda self, observations: [START])
    assert replay_check.replay_episode(runs[row] / EPISODE, meta, tmp_path / "start_alone", store, world)["problems"] != []


def test_only_the_server_and_the_yardstick_look_blocks_up():
    """Models.served_blocks, the lookup of released readings in the models file, is called by the server's belief and by the
    yardstick that replays it, and by no other module."""
    package = Path(ROOT) / "CausalDiagnosis"
    calling = {p.relative_to(package).as_posix() for p in package.rglob("*.py")
               if "tests" not in p.relative_to(package).parts and "debug" not in p.relative_to(package).parts
               and "served_blocks(" in p.read_text(encoding="utf-8", errors="replace")}
    assert calling == {"mcp_servers/Causal_Diagnosis_MCP/tools/fitted_models.py", "mcp_servers/Causal_Diagnosis_MCP/tools/causal_tools.py",
                       "evaluation/metrics/realized_read_elimination.py"}
