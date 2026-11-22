"""The parts of the run grid, the reference lines of the ablations and the switch of ablation Ag9."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.experiments.table_definitions import experiment_tables as design  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs import grid_files, grid_job_lists as grid_lines, run_lines  # noqa: E402


def test_every_place_of_the_grid_reads_a_cell_another_line_plays_so_no_reference_line_is_written():
    assert not [c for c, _l in run_lines.ablations() if ", the reference, " in c]
    for preset, rows in design.ABLATION_PLACES.values():
        assert all((preset, row) in design.played_cells() for row in rows)


def test_an_ablation_on_a_preset_its_row_does_not_play_gets_its_reference_line(monkeypatch):
    priced = run_lines.run_line(design.PRICED_PRESET, design.CD_MCP)
    cells = design.played_cells() - {(design.PRICED_PRESET, design.CD_MCP)}
    monkeypatch.setattr(design, "played_cells", lambda: set(cells))
    reference = [(c, l) for c, l in run_lines.ablations() if ", the reference, " in c]
    assert [l for _, l in reference] == [priced]  # once, before the first arm of Ag7, the one ablation of the place
    first_arm = next(i for i, (c, _) in enumerate(run_lines.ablations()) if c.startswith("Ag7, the agent's own"))
    assert run_lines.ablations()[first_arm - 1][1] == priced


def test_ag9_writes_one_line_set_on_the_preset_of_training_and_no_second_reference_line():
    lines = [l for c, l in run_lines.ablations() if c.startswith(f"{design.PARALLEL_ABLATION},")]
    assert lines == [run_lines.run_line(design.TRAINING_PRESET, design.CD_MCP, "--causal-sequential")]
    assert (design.TRAINING_PRESET, design.CD_MCP) in design.played_cells()


def test_the_parts_cover_the_grid_file_and_name_their_replicates():
    assert run_lines.PARTS == ("main", "companion", "ladder", "layers", "training", "ablations", "pilot", "gate", "sampling",
                                "environment_token", "layers_token", "misdiagnosis_price", "environment", "follower")
    agents = (run_lines.part("main") + run_lines.part("companion") + run_lines.part("ladder") + run_lines.part("ablations")
              + run_lines.part("environment_token") + run_lines.part("layers_token") + run_lines.part("misdiagnosis_price")
              + run_lines.part("environment") + run_lines.part("follower"))
    grid_run_lines = [line for line in grid_files.files()["grid_agents.txt"].splitlines() if not line.startswith("#")]
    assert grid_run_lines == [line for _, line in agents]
    assert len(run_lines.part("main")) == len(design.AGENT_ROWS) + len(design.HEALTHY_CONTROL_ROWS)
    assert all(run_lines.is_companion("En9", line) for _, line in run_lines.part("companion"))
    assert run_lines.replicates("main") == len(run_lines.pending_lines.REPLICATES) and run_lines.replicates("pilot") == 1


def test_the_part_layers_plays_the_rungs_between_layer_0_and_the_full_environment_alone():
    """The layer study for a job of its own: the rungs above layer 0 and below all six big layers with the three designs with the
    CausalDiagnosis MCP, one run per replicate each, which part ladder holds too beside the three rows without the MCP on all six big
    layers, the baselines of the training table, while the agents table and the part environment play the first and the top rung."""
    middle = [preset for preset in design.LADDER if preset not in (design.AGENTS_PRESET, design.TRAINING_PRESET)]
    lines = run_lines.part("layers")
    assert [line for _, line in lines] == [run_lines.run_line(preset, row) for preset in middle for row in design.LADDER_ROWS]
    assert len(lines) == len(middle) * len(design.LADDER_ROWS) and run_lines.replicates("layers") == run_lines.replicates("ladder")
    top = [entry for entry in run_lines.part("ladder") if entry not in lines]
    assert [line for _, line in top] == [run_lines.run_line(design.TRAINING_PRESET, row) for row in design.TRAINING_BASELINES]


def test_the_token_parts_play_the_layers_and_the_six_rows_on_the_token_twins_and_nothing_else():
    """Part layers_token is part layers with every rung that has a token twin on its twin, and part environment_token the six rows
    of the agents on the twin of all six big layers, both one run per replicate, and every twin differs from its preset by the clock."""
    twins = run_lines.token_twins()
    layers = [line for _, line in run_lines.part("layers")]
    on_twins = [line.replace(f"--config {preset} ", f"--config {twins[preset]} ") for line in layers
                for preset in [line.split()[1]] if preset in twins]
    assert [line for _, line in run_lines.part("layers_token")] == on_twins and on_twins
    assert [line for _, line in run_lines.part("environment_token")] == [run_lines.run_line(twins[design.TRAINING_PRESET], row)
                                                                          for row in design.AGENT_ROWS]
    assert run_lines.replicates("layers_token") == run_lines.replicates("environment_token") == run_lines.replicates("layers")
    assert set(twins) == {preset for preset in design.LADDER if preset in design.TOKEN_PRESETS.values()}


def test_the_reference_runs_of_a_part_print_as_grid_lines_that_pending_lines_reads(capsys):
    """Every reference agent on every preset of a part, as a grid file, so that pending_grid_lines.py checks the reference runs of a part as
    it checks its model lines: here the layer study, whose presets set their own clocks, and a line with a clock mode of its own."""
    middle = [preset for preset in design.LADDER if preset not in (design.AGENTS_PRESET, design.TRAINING_PRESET)]
    references = grid_files.reference_lines(run_lines.part("layers"))
    assert [line for _, line in references] == [f"--config {preset} --agent {agent}" for preset in middle for agent in run_lines.REFERENCE_AGENTS]
    fields = run_lines.pending_lines.expected(references[-1][1], "sampled")
    assert (fields["config_name"], fields["agent"]) == (middle[-1], run_lines.REFERENCE_AGENTS[-1])
    clocked = grid_files.reference_lines([("a line", f"--config {middle[0]} --clock-mode instant --agent causal")])
    assert [line for _, line in clocked] == [f"--config {middle[0]} --clock-mode instant --agent {agent}" for agent in run_lines.REFERENCE_AGENTS]
    assert grid_lines.main(["references", "layers"]) == 0
    printed = [line for line in capsys.readouterr().out.splitlines() if not line.startswith("#")]
    assert printed == [line for _, line in references]


def test_the_part_command_prints_a_part_under_a_header_that_names_its_runs_per_line(capsys):
    """The grid file of one part, as a job of CausalDiagnosis/cluster_jobs/3_run/run_lines.sh writes it for GRID_PART: a header that names the part and the runs
    of each line, then the lines of the part in their order, for a part of the test set and one of the development starts."""
    for name in ("main", "gate"):
        assert grid_lines.main(["part", name]) == 0
        out = capsys.readouterr().out
        assert out.startswith(f"# the part {name} of the run grid, {run_lines.replicates(name)} run(s) per line")
        assert [line for line in out.splitlines() if not line.startswith("#")] == [line for _, line in run_lines.part(name)]


def test_an_unknown_part_or_row_exits_2_and_prints_nothing(capsys):
    assert grid_lines.main(["part", "nosuch"]) == 2 and grid_lines.main(["row", "nosuch"]) == 2 and grid_lines.main(["references", "nosuch"]) == 2
    assert capsys.readouterr().out == ""
    assert grid_lines.main(["row", "causal+mcp"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == run_lines.run_line(design.AGENTS_PRESET, design.CD_MCP)


def test_no_grid_file_names_a_decision_id_or_a_table_number_of_the_paper():
    import re
    for name, text in grid_files.files().items():
        assert not re.search(r"\bD\d{2,3}\b|\bTable \d", text), name

