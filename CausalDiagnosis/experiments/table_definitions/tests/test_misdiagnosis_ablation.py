"""The ablation L1 of the misdiagnosis price at the budget: the small switch misdiagnosis of big layer 1 in its mode budget sets the
price of a wrong accepted diagnosis to the budget of the asset, read from the asset file and typed nowhere, and the preset of the arm
is the preset of training with that one setting. The arm plays the three designs without the MCP in the place of the switches of
layers 2, 3 and 5, and its table describes both prices with no test, while SWITCHES keeps the switches of layers 2, 3 and 5."""
from CausalDiagnosis.utils.paths import PACKAGE
import json
import shutil
from pathlib import Path

import pytest

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.experiments.cluster_runs import grid_files, run_lines
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config
from CausalDiagnosis.datasets.tests.conftest import mount_port
from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import BASE_META, _base_row, _run
from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs_registered import CTX

ARM = next(preset for preset, (ablation, _) in design.ARM_PRESETS.items() if ablation == "L1")
PORT = PACKAGE / "experiments" / "episode_runners" / "tests" / "ports" / "bench_1"


def test_the_misdiagnosis_price_at_the_budget_is_the_budget_of_the_asset(tmp_path):
    """The arm composes with the misdiagnosis price equal to the budget of the asset on the rig and on bench_1, and with every
    other value of the preset of training; it is the one arm of L1, placed as the switches of layers 2, 3 and 5, and the grid plays
    it once per design without the MCP in the part misdiagnosis_price, which grid_agents.txt plays after every earlier line and
    before the part environment alone, the part appended after it."""
    for asset_id in ("hyd_1", "bench_1"):
        shutil.copytree(PORT, tmp_path / PORT.name, dirs_exist_ok=True, ignore=shutil.ignore_patterns("data", "__pycache__"))
        with mount_port(tmp_path):
            _name, arm = load_config(ARM, asset_id)
            _name, training = load_config(design.TRAINING_PRESET, asset_id)
        a, t = arm.to_json(), training.to_json()
        assert a["budget"] and a["prices"]["misdiagnosis"] == a["budget"] != t["prices"]["misdiagnosis"]
        assert {**a, "prices": {**a["prices"], "misdiagnosis": t["prices"]["misdiagnosis"]}} == t
    assert [ablation for ablation, _ in design.ARM_PRESETS.values()].count("L1") == 1
    assert design.ABLATION_PLACES["L1"] == design.ABLATION_PLACES["L2"] == (design.TRAINING_PRESET, design.FRAMEWORK_ROWS)
    lines = [line for _, line in run_lines.part("misdiagnosis_price") if f"--config {ARM} " in f"{line} "]
    assert len(lines) == len(design.FRAMEWORK_ROWS) and not [line for _, line in run_lines.part("ablations") if f"--config {ARM} " in f"{line} "]
    grid_run_lines = [line for line in grid_files.files()["grid_agents.txt"].splitlines() if not line.startswith("#")]
    environment = [line for _, line in run_lines.part("environment")]
    assert grid_run_lines[-len(lines) - len(environment):] == lines + environment  # appended in order, so each earlier line keeps its place


def _episode(run_dir, seed, charge, reads):
    """The hidden files of one episode: a verdict that names the misdiagnosis charge when there is one, and a log whose reads before
    the diagnosis serve the given cycles and whose read after it serves cycles that no count may take."""
    hidden = run_dir / f"seed{seed:06d}" / "hidden"
    hidden.mkdir(parents=True, exist_ok=True)
    (hidden / "verdict.json").write_text(json.dumps({"misdiagnosis_charge": charge} if charge else {}), encoding="utf-8")
    log = [{"event": "reset"}, *({"event": "read", "cycles": cycles} for cycles in reads), {"event": "submit_diagnosis"}, {"event": "read", "cycles": 9}]
    (hidden / "log.json").write_text(json.dumps(log), encoding="utf-8")


def test_the_table_of_l1_describes_both_prices_with_no_test_and_switches_leaves_it_out(tmp_path):
    """MISDIAGNOSIS_PRICE: one row per design and preset, the charge of the verdicts, the cost without it on one scale for both
    presets, the cycles read before the first diagnosis, no difference, no p value and no Holm cell; SWITCHES reads no episode of
    the arm."""
    runs = tmp_path / "runs"
    for preset, charges in ((design.TRAINING_PRESET, (0, 0, 0, 0)), (ARM, (40, 0, 40, 0))):
        mode = aggregate.config_clock_mode(preset)
        rows = [_base_row(s, agent="plan_execute_m", diagnostics_server=False, config=preset, clock_mode=mode, cost=10 + charges[s],
                          diagnosis_all_correct=not charges[s], abstained=False) for s in range(4)]
        _run(runs, preset, rows, {**BASE_META, "config_name": preset})
        for s in range(4):
            _episode(runs / preset, s, charges[s], (3, 2))
    episodes = aggregate.add_derived_fields(aggregate.load_episodes(runs))
    found = {row["config"]: row for row in aggregate.table_misdiagnosis_price(episodes, {**CTX, "resamples": 20})}
    assert set(found) == {design.TRAINING_PRESET, ARM} and all(row["n"] == 4 for row in found.values())
    assert (found[ARM]["misdiagnosis_charge"], found[design.TRAINING_PRESET]["misdiagnosis_charge"]) == (20.0, 0.0)
    assert found[ARM]["cost_without_charge"] == found[design.TRAINING_PRESET]["cost_without_charge"] == 10.0
    assert found[ARM]["cycles_read_before_diagnosis"] == 5.0
    assert not any(key.endswith(("_p", "_diff", "_holm_p", "_reject")) for row in found.values() for key in row)
    assert not [e for e in aggregate.switch_episodes(episodes) if e["config"] == ARM]
