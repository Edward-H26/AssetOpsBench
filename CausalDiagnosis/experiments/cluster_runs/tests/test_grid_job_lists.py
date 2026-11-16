"""The grid files of CausalDiagnosis/cluster_jobs/ against the design they are written from: experiment_tables.py for the rows, the presets and the places of the
supporting ablations, training_variants/ for the variants and the folds of the training table, and the presets of the configs folder."""
from CausalDiagnosis.utils.paths import CLUSTER_JOBS, preset_files
import json
import re

import pytest

from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.experiments.cluster_runs import grid_files, grid_job_lists as grid_lines, run_lines
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines
from CausalDiagnosis.experiments.cluster_runs import precompute_grid_tables as precompute_tables
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.experiments.table_definitions import training_variants as training
from CausalDiagnosis.experiments.episode_runners.run_configuration import CONFIGS, load_config

START_SETS = {"grid_agents.txt": "sampled", "grid_pilot.txt": "combinations", "grid_gate.txt": "combinations",
              "grid_training_sampling.txt": "training", "grid_training_evaluation.txt": "sampled"}


def _lines(name):
    return grid_files.grid_lines(grid_files.HPC / name)


def _played(name):
    """(preset, agent, with the MCP) of every line of a grid file, as run_episodes/ resolves it."""
    fields = [pending_lines.expected(line, START_SETS[name]) for line in _lines(name)]
    return [(f["config_name"], f["agent"], f["diagnostics_server"]) for f in fields]


def _row(row):
    return row[0], row[1]


def test_every_grid_file_of_hpc_is_the_one_the_design_implies():
    """No run line of the cluster is written by hand: every grid file equals what grid_job_lists.py writes from experiment_tables.py and
    training_variants/, and CausalDiagnosis/cluster_jobs/ holds no other grid file."""
    assert grid_files.differing() == []
    assert sorted(p.name for p in grid_files.HPC.glob("grid_*.txt")) == sorted(grid_files.FILES) == sorted(grid_files.files()) == sorted(START_SETS)


def test_the_first_lines_are_the_six_rows_of_the_agents_table_on_layer_0():
    """The six rows play layer 0 alone, and they open the file, so a cut of its first lines is the agents table."""
    rows = len(design.AGENT_ROWS)
    assert _played("grid_agents.txt")[:rows] == [(design.AGENTS_PRESET, *_row(row)) for row in design.AGENT_ROWS]
    assert rows == 6 and design.AGENTS_PRESET == "base"


def test_every_rung_of_the_ladder_is_played_once_by_the_three_agents_with_the_mcp():
    """The environment table plays every preset of design.LADDER with the three agents with the MCP, each run once, so the base
    rung is three lines of the agents table and the top rung the part environment, and the three rows without the MCP on all six
    big layers, the baselines of the training table, follow the middle rungs as lines of their own, each once."""
    lines = _lines("grid_agents.txt")
    for preset in design.LADDER:
        for row in design.LADDER_ROWS:
            assert lines.count(run_lines.run_line(preset, row)) == 1, (preset, row[2])
    assert all(with_mcp for _agent, with_mcp, _name in design.LADDER_ROWS) and len(design.LADDER) == 6
    assert not any(with_mcp for _agent, with_mcp, _name in design.TRAINING_BASELINES)
    baselines = [run_lines.run_line(design.TRAINING_PRESET, row) for row in design.TRAINING_BASELINES]
    assert baselines == [line for _comment, line in run_lines.ladder()[-3:]] and all(lines.count(line) == 1 for line in baselines)
    assert len(lines) == len(set(lines)) == 100


def test_every_supporting_ablation_plays_its_arms_on_the_preset_and_with_the_rows_of_its_place():
    """Each supporting ablation of design.ABLATION_PLACES has one line per arm and row, an arm that is a preset of
    design.ARM_PRESETS on that preset and every other arm on the preset of the ablation's place. The parts ablations and companion
    play every arm once in the order of the places, the companion the arms whose switch grid_lines.COMPANION_SWITCHES names, and
    their other lines are the reference lines of the places whose rows play nowhere else."""
    for name, (preset, rows) in design.ABLATION_PLACES.items():
        entries = run_lines.ablation(name)
        own = [arm_preset for arm_preset, (owner, _arm) in design.ARM_PRESETS.items() if owner == name]
        arms = len(run_lines.SWITCH_ARMS.get(name, ())) + len(own)
        if name in run_lines.NO_LINE:
            assert entries == []
        elif name in run_lines.AGENT_ARMS:
            assert [line for _comment, line in entries] == [f"--config {preset} {run_lines.SERVERS} --agent {run_lines.AGENT_ARMS[name][1]}"]
        else:
            played = [(f["config_name"], f["agent"], f["diagnostics_server"]) for f in (pending_lines.expected(line) for _comment, line in entries)]
            assert arms and len(entries) == arms * len(rows) and {(agent, with_mcp) for _preset, agent, with_mcp in played} == {_row(row) for row in rows}
            assert {played_preset for played_preset, _agent, _mcp in played} == ({preset} if name in run_lines.SWITCH_ARMS else set(own))
        assert all(comment.startswith(f"{name}, ") for comment, _line in entries)
    arms = [line for name in design.ABLATION_PLACES if name not in run_lines.APPENDED_ABLATIONS for _comment, line in run_lines.ablation(name)]
    assert run_lines.part("misdiagnosis_price") == [entry for name in run_lines.APPENDED_ABLATIONS for entry in run_lines.ablation(name)]
    companion, rest = [line for _comment, line in run_lines.companion()], run_lines.ablations()
    assert len(set(arms)) == len(arms) and [line for _comment, line in rest if line in arms] == [line for line in arms if line not in companion]
    assert companion == [line for name, switch in run_lines.COMPANION_SWITCHES.items() for _comment, line in run_lines.ablation(name)
                         if line.endswith(f" {switch}")]
    assert all(f", {design.REFERENCE_ARM}, " in comment for comment, line in rest if line not in arms)


def test_an_ablation_without_arms_and_an_arm_without_a_place_stop_the_generator(monkeypatch):
    monkeypatch.setitem(design.ABLATION_PLACES, "En99", (design.AGENTS_PRESET, design.MCP_ROWS))
    with pytest.raises(ValueError, match="En99 of design.ABLATION_PLACES has no arm"):
        run_lines.ablations()
    monkeypatch.delitem(design.ABLATION_PLACES, "En99")
    monkeypatch.delitem(design.ABLATION_PLACES, "En5")
    with pytest.raises(ValueError, match="gives no place to En5"):
        run_lines.ablations()


def test_every_line_of_the_ablations_is_an_arm_of_its_ablation_on_the_leaderboard():
    """The switch of an arm is the one the leaderboard reads as that ablation: every line of the supporting ablations, resolved
    as run_episodes/ resolves it, lands in the slot of the ablation its comment names."""
    from CausalDiagnosis.evaluation.result_tables import leaderboard

    settings = ("clock_mode", "guidance_level", "verifier_scope", "ranking_model", "prices", "diagnostics_server", "abstention_rule")
    for comment, line in run_lines.ablations():
        fields = pending_lines.expected(line)
        label = run_episodes.agent_label_of(fields["agent"], "m", fields.get("causal_options"))
        run = {k: str(v) for k, v in {"run": "x", "mode": "gym", "agent": label, "model": "m", "config": fields["config_name"],
                                      "training_variant": training.AS_RELEASED, "started": "", **{k: fields[k] for k in settings}}.items()}
        slot = leaderboard.classify(run)
        assert slot is not None and slot.kind == "ablation" and slot.ablation == comment.split(",")[0], (comment, slot)


def test_the_time_pressure_curve_plays_half_and_twice_the_thinking_rate_with_the_three_agents_without_the_mcp():
    """The presets that scale think_seconds_per_cycle of the asset file, at 0.5 and at 2 and never at a number
    of seconds, are played by the three agents without the MCP, six lines beside the thinking time off."""
    scaled = {}
    for path in sorted(preset_files(CONFIGS).values()):
        rate = json.loads(path.read_text(encoding="utf-8")).get("switches", {}).get("think_seconds_per_cycle")
        if isinstance(rate, dict):
            scaled[path.stem] = rate["scale"]
    assert sorted(scaled.values()) == [0.5, 2]
    played = [entry for entry in _played("grid_agents.txt") if entry[0] in scaled]
    assert sorted(played) == sorted((preset, *_row(row)) for preset in scaled for row in design.FRAMEWORK_ROWS) and len(played) == 6


def test_every_preset_of_the_configs_folder_is_played_or_named_as_left_out():
    """A preset that lands in the configs folder is either played by a line of the grid or named by experiment_tables.py as a preset that
    no table reads or that the paper reports apart, so no new preset stays without its lines. The grid plays the rungs of the
    ladder, the presets of the ablation arms and the healthy control, which a table of its own reports."""
    played = {preset for preset, _agent, _mcp in _played("grid_agents.txt")}
    assert played | set(design.UNREPORTED) | set(design.OWN_TABLES) == {path.stem for path in preset_files(CONFIGS).values()}
    assert played == set(design.LADDER) | set(design.ARM_PRESETS) | {design.HEALTHY_CONTROL_PRESET} | set(design.TOKEN_PRESETS)


def test_the_pilot_and_the_gate_play_development_runs_of_the_rows_their_rules_name():
    """The pilot of the thinking rate plays the three agents with the MCP on the full environment, and the development gate
    the CausalDiagnosis agent and the paper-era Plan-Execute without the MCP on layer 0."""
    assert _played("grid_pilot.txt") == [(design.TRAINING_PRESET, *_row(row)) for row in design.TRAINING_ROWS]
    assert _played("grid_gate.txt") == [(design.AGENTS_PRESET, "causal", False), (design.AGENTS_PRESET, "plan_execute", False)]
    assert all(pending_lines.expected(line, "combinations")["starts"] == "combinations" for name in ("grid_pilot.txt", "grid_gate.txt") for line in _lines(name))


def test_the_training_table_samples_every_agent_and_fold_and_evaluates_every_variant():
    """One sampling run per agent, fold and sample of the registered count, the agent without
    training in one evaluation line and every trained variant in one line per fold, and one loop of GEPA per block a variant plays."""
    agents = [row[0] for row in design.TRAINING_ROWS]
    assert tuple(agents) == training.TRAINED_AGENTS and all(with_mcp for _agent, with_mcp, _name in design.TRAINING_ROWS)
    samples = range(1, training.CONFIG["samples_per_state"] + 1)
    sampled = [pending_lines.expected(line, "training") for line in _lines("grid_training_sampling.txt")]
    assert [(f["agent"], f["training_fold"], f["training_sample"]) for f in sampled] == [(a, fold, k) for a in agents for fold in training.TRAINING_FOLDS for k in samples]
    assert {(f["config_name"], f["training_variant"], f["diagnostics_server"]) for f in sampled} == {(design.TRAINING_PRESET, training.AS_RELEASED, True)}
    evaluated = [pending_lines.expected(line) for line in _lines("grid_training_evaluation.txt")]
    trained = [variant for variant in training.TRAINING_VARIANTS if variant != training.AS_RELEASED]
    wanted = [(a, variant, fold) for a in agents for variant, fold in
              [(training.AS_RELEASED, None)] + [(variant, fold) for variant in trained for fold in training.TRAINING_FOLDS]]
    assert [(f["agent"], f["training_variant"], f["training_fold"]) for f in evaluated] == wanted
    assert {(f["config_name"], f["diagnostics_server"]) for f in evaluated} == {(design.TRAINING_PRESET, True)}


def test_the_second_round_of_sampling_plays_the_next_samples_of_one_agent_and_fold(capsys):
    """An agent and fold whose first round kept too few episodes samples once more, the next samples of the
    training configuration for that agent and fold alone, which the command writes as a grid file and never by hand."""
    agent, fold = design.CD[0], training.TRAINING_FOLDS[-1]
    entries = run_lines.second_round(agent, fold)
    fields = [pending_lines.expected(line, "training") for _comment, line in entries]
    assert [(f["agent"], f["training_fold"], f["training_sample"]) for f in fields] == [(agent, fold, k) for k in training.extension_samples(training.DEFAULT_ASSET)]
    assert fields and min(f["training_sample"] for f in fields) == training.CONFIG["samples_per_state"] + 1
    assert {(f["config_name"], f["training_variant"], f["diagnostics_server"]) for f in fields} == {(design.TRAINING_PRESET, training.AS_RELEASED, True)}
    assert all(comment.startswith("training table, second round of sampling, ") and f"fold {fold}" in comment for comment, _line in entries)
    for wrong in (("nobody", fold), (agent, max(training.TRAINING_FOLDS) + 1)):
        with pytest.raises(SystemExit):
            run_lines.second_round(*wrong)
    assert grid_lines.main(["second_round", agent, str(fold)]) == 0
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if not line.startswith("#")] == [line for _comment, line in entries]
    assert out.startswith(f"# the second round of sampling for {agent} on fold {fold}") and "GRID_STARTS=training" in out


@pytest.mark.parametrize("name", sorted(START_SETS))
def test_every_line_parses_under_its_start_set_and_no_two_lines_play_one_condition(name):
    """A second finished run of one condition is refused by every table, so no file holds two lines that resolve to the same
    run, and the tables of every line are computed before a job starts, so every line resolves for precompute_grid_tables.py too."""
    lines = _lines(name)
    conditions = [json.dumps(pending_lines.expected(line, START_SETS[name]), sort_keys=True) for line in lines]
    assert len(set(conditions)) == len(conditions) == len(lines) > 0
    assert len(precompute_tables.configurations([grid_files.HPC / name], [])) == len(lines)


def test_the_header_of_every_file_states_the_counts_of_its_lines():
    headers = {}
    for name, text in grid_files.files().items():
        lines = text.splitlines()
        first_run = next(i for i, line in enumerate(lines) if not line.startswith("#"))
        headers[name] = " ".join(line[2:] for line in lines[:first_run - 1])
        assert all(len(line) <= 120 for line in lines[:first_run - 1]), name
        runs = len(_lines(name))
        stated = [int(n) for n in re.findall(r"(\d+) run lines", headers[name])]
        assert stated in ([runs], []) and (stated or runs <= 3), name
    runs, healthy = len(_lines("grid_agents.txt")), len(run_lines.healthy_control())
    assert f"{runs} run lines, one run per line and replicate of GRID_REPLICATES, {run_lines.REPLICATES} replicates" in headers["grid_agents.txt"]
    assert (f"{run_lines.STARTS} episodes per run on a preset with faulty starts and {run_lines.HEALTHY_STARTS} on the healthy control "
            f"({((runs - healthy) * run_lines.STARTS + healthy * run_lines.HEALTHY_STARTS) * run_lines.REPLICATES} episodes)") in headers["grid_agents.txt"]
    assert (f"Lines 1 to {len(design.AGENT_ROWS)} are the main table, whose runs the rescoring of En4 scores again, and the next {healthy} "
            "the healthy control on the start with no faulty component") in headers["grid_agents.txt"]
    assert run_lines.STARTS == 47 and run_lines.REPLICATES == len(pending_lines.REPLICATES) == 5


def test_the_reference_runs_follow_the_presets_of_a_grid_file_and_the_policies_of_the_design(tmp_path):
    """run_grid.sbatch plays the reference policies of experiment_tables.py, with the scripted agent of the wiring check, on the presets
    that the lines of its grid file name, each once in the order the file first names it (layer 0 and the healthy control of the
    part main, the rungs of the ladder, then the presets of the ablation arms), a clock mode a line overrides after a colon."""
    assert run_lines.REFERENCE_AGENTS == (*(label for label, _name in design.REFERENCE_POLICIES), "scripted")
    assert set(run_lines.REFERENCE_AGENTS) <= set(run_episodes.AGENTS)
    assert grid_files.presets(CLUSTER_JOBS / "grids" / "grid_agents.txt") == list(dict.fromkeys(
        [design.AGENTS_PRESET, design.HEALTHY_CONTROL_PRESET, *design.LADDER, *(preset for preset, _agent, _mcp in _played("grid_agents.txt"))]))
    grid = tmp_path / "local.txt"
    grid.write_text("# a comment\n--config base --agent causal\n--config timed --clock-mode instant --agent causal\n--config base --agent static\n", encoding="utf-8")
    assert grid_files.presets(grid) == ["base", "timed:instant"]


def test_the_lines_of_a_grid_file_split_by_whether_their_preset_charges_the_wall_clock(tmp_path, capsys):
    """A run whose clock charges the wall seconds of thinking plays on the node and with the runs in flight
    of the pilot, so the lines of a grid file split into those of the generation time clock and the others, which run no such
    clock: layer 0 and the healthy control on it, the priced rung and the thinking time off. A line that overrides its clock mode
    follows the override."""
    lines = _lines("grid_agents.txt")
    clocked, free = (grid_files.by_clock(CLUSTER_JOBS / "grids" / "grid_agents.txt")[key] for key in (True, False))
    assert sorted(clocked + free) == sorted(lines) and clocked and free and free[:6] == lines[:6]
    presets = {pending_lines.expected(line)["config_name"] for line in free}
    assert presets == {name for name in (*design.LADDER, *design.ARM_PRESETS, design.HEALTHY_CONTROL_PRESET, *design.TOKEN_PRESETS)
                       if load_config(name)[1].clock_mode != run_lines.WALL_CLOCK}
    assert {"base", "priced"} <= presets and "deterioration" not in presets
    for name, wanted in (("grid_pilot.txt", True), ("grid_training_sampling.txt", True), ("grid_training_evaluation.txt", True), ("grid_gate.txt", False)):
        assert grid_files.by_clock(grid_files.HPC / name)[not wanted] == [], name
    grid = tmp_path / "local.txt"
    grid.write_text("--config base --agent causal\n--config base --clock-mode generation_time --agent causal\n", encoding="utf-8")
    assert grid_lines.main(["clocked", str(grid)]) == 0 and capsys.readouterr().out == "--config base --clock-mode generation_time --agent causal\n"
    assert grid_lines.main(["unclocked", str(grid)]) == 0 and capsys.readouterr().out == "--config base --agent causal\n"
    assert grid_lines.main(["clocked", str(CLUSTER_JOBS / "grids" / "grid_gate.txt")]) == 0 and capsys.readouterr().out == ""


def _finished(root, name, line, **change):
    """A finished run folder of a grid line of the test set, with the run.json fields that pending_grid_lines.py matches."""
    folder = root / name
    folder.mkdir(parents=True)
    meta = {**pending_lines.expected(line), "started": "2026-01-01T00:00:00", "ended": "2026-01-01T01:00:00", "episodes_planned": 1, **change}
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    (folder / "scores.csv").write_text("seed\n0\n", encoding="utf-8")
    return folder


def test_the_runs_of_the_agents_table_are_found_by_what_they_played_whatever_file_their_job_read(tmp_path, capsys):
    """The rescoring of En4 scores the finished runs of the six rows again, every replicate, found by the settings of their
    run.json and never by the index of a grid line, so a run that a job played from a continuation file is found, and neither
    an arm on layer 0, a rung of the ladder, a rescored copy nor an unfinished run is."""
    rows = [line for _comment, line in run_lines.agents()]
    found = [_finished(tmp_path, "pe_r1", f"{rows[0]} --replicate 1"), _finished(tmp_path, "pe_r4", f"{rows[0]} --replicate 4"),
             _finished(tmp_path, "cd_mcp_r5", f"{rows[5]} --replicate 5")]
    _finished(tmp_path, "arm", f"{rows[5]} --guidance 1")
    _finished(tmp_path, "rung", run_lines.run_line(design.LADDER[1], design.PE))
    _finished(tmp_path, "pe_r1_scope_all", f"{rows[0]} --replicate 1", verifier_scope="all")
    _finished(tmp_path, "running", f"{rows[1]} --replicate 1", ended=None)
    assert grid_files.agent_runs([tmp_path]) == sorted(found)
    assert grid_lines.main(["agents", str(tmp_path)]) == 0 and capsys.readouterr().out.splitlines() == [str(folder) for folder in sorted(found)]


def test_the_runs_of_table_1_take_the_copy_scored_again_under_its_own_scope_in_place_of_its_run(tmp_path):
    """After a fault of class scoring the rescoring of En4 reads what the tables read: a run that rescore_under_verifier_scope.py --own-scope
    scored again under the roots gives way to its copy, and a run without such a copy stays."""
    rows = [line for _comment, line in run_lines.agents()]
    _finished(tmp_path, "pe_r1", f"{rows[0]} --replicate 1")
    copy = _finished(tmp_path, "pe_r1_own_scope", f"{rows[0]} --replicate 1", rescored={"from": "pe_r1", "own_scope": True})
    kept = _finished(tmp_path, "pe_r2", f"{rows[0]} --replicate 2")
    assert grid_files.agent_runs([tmp_path]) == sorted([copy, kept])


def test_the_folds_command_prints_the_folds_of_the_registered_split(capsys):
    """CausalDiagnosis/cluster_jobs/3_run/run_gepa_online.sh loops over the folds this command prints, so no fold list is typed in a job script."""
    assert grid_lines.main(["folds"]) == 0
    assert capsys.readouterr().out.split() == [str(fold) for fold in training.held_out_folds.fold_ids()]


def test_the_wiring_command_prints_the_agents_of_the_wiring_check(capsys):
    """run_grid.sbatch reads the agents of the wiring check from this command, the agents of pending_lines.WIRING_AGENTS, so
    the runner and the count of pending pairs play such an agent on the first replicate alone; each is a reference agent."""
    assert grid_lines.main(["wiring"]) == 0
    assert capsys.readouterr().out.split() == list(pending_lines.WIRING_AGENTS)
    assert set(pending_lines.WIRING_AGENTS) <= set(run_lines.REFERENCE_AGENTS)


def test_the_command_checks_writes_and_prints(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(grid_files, "HPC", tmp_path)
    assert grid_lines.main([]) == 1 and capsys.readouterr().out.count("differs from the design") == len(grid_files.FILES)
    assert grid_lines.main(["write"]) == 0 and grid_lines.main(["check"]) == 0
    capsys.readouterr()
    (tmp_path / "grid_gate.txt").write_text("--config base --agent causal\n", encoding="utf-8")
    assert grid_lines.main([]) == 1 and capsys.readouterr().out.startswith("CausalDiagnosis/cluster_jobs/grids/grid_gate.txt differs from the design")
    assert grid_lines.main(["policies"]) == 0 and capsys.readouterr().out.split() == list(run_lines.REFERENCE_AGENTS)
    assert grid_lines.main(["presets", str(tmp_path / "grid_gate.txt")]) == 0 and capsys.readouterr().out == "base\n"


def test_no_grid_file_co_serves_the_judge():
    """No serving job of a grid co-serves the judge: the judge runs after the grid over the folders of its runs in a serving job of
    the judge alone, so every line of every table plays under one serving stack, and the header of the evaluation runs of the
    training table says so."""
    texts = grid_files.files()
    assert not [name for name, text in texts.items() if "co-served" in text or "CO_SERVE_JUDGE" in text]
    header = " ".join(line.removeprefix("#") for line in texts["grid_training_evaluation.txt"].splitlines() if line.startswith("#"))
    assert "in a serving job of the judge alone (run_judge.sh)" in " ".join(header.split())


def test_no_registered_line_names_the_threshold_card():
    """No switch of the agent that no registered run sets stays in the fork: no line that the generator writes for a grid file
    holds the option of the threshold card, and the driver refuses the option, so no run can set the card."""
    for name, text in grid_files.files().items():
        assert "--threshold-card" not in text, name
    with pytest.raises(SystemExit):
        run_episodes.build_parser().parse_args(["--agent", "causal", "--threshold-card", "card.json"])


def test_the_runs_of_the_agents_table_are_found_on_the_model_the_job_served(tmp_path, capsys, monkeypatch):
    """A run of a row records the model its job served, which no grid line names, so the agents query takes it as --model."""
    rows = [line for _comment, line in run_lines.agents()]
    monkeypatch.setattr(pending_lines, "tree_identity", lambda agent: "a" * 64)
    fields = pending_lines.expected(rows[0])
    servers = ["iot", "fmsr", "wo", "task"] if fields["surface"] == "aob" else ["hydraulic_rig"]
    backend = {"kind": fields["backend_kind"]} if fields["backend_kind"] else None
    served = _finished(tmp_path, "pe_r1", f"{rows[0]} --replicate 1", model="served-model", servers=servers, backend=backend,
                       code={"identity": "a" * 64})
    assert grid_files.agent_runs([tmp_path]) == [] and grid_files.agent_runs([tmp_path], "served-model") == [served]
    assert grid_lines.main(["agents", "--model", "served-model", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == [str(served)]
