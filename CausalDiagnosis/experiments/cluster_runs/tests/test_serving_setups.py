"""The two cells of every registered comparison play on one serving setup, the number of LoRA adapters that the server of the agent
model holds beside it (run.json lora_adapters, from the endpoint file of the serving job). The grid files place every cell of a
registered family on the server of grid_agents.txt but the cells of the training family, which grid_training_evaluation.txt plays
beside the adapters; the page starts a server with the adapters for the part training alone; a run on each setup is a condition of its
own; and the tables read each comparison on one setup. The image of a run of record is that of the serving file of record and its GPU
card that of record when env.sh names one, which registration() checks, and the group key of the tables holds both."""
from CausalDiagnosis.utils.paths import CLUSTER_JOBS
import re
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs import grid_files, run_lines  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import training_variants as training  # noqa: E402
from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import _run  # noqa: E402
from CausalDiagnosis.evaluation.result_tables.tests.test_paper_tables import COMBINATIONS, MODEL, SEEDS, _aggregated, _episode, _lines, _meta  # noqa: E402

README = (CLUSTER_JOBS / "README.md").read_text(encoding="utf-8")
SEQUENTIAL = dict(run_lines.SWITCH_ARMS[design.PARALLEL_ABLATION])  # the arm of Ag9 by its switch, the side of the pair that runs it off


def _file_lines(name):
    return [line for line in grid_files.files()[name].splitlines() if not line.startswith("#")]


def _registered_cells():
    """{family: [(run line, run line)]} of every registered family but training: the lines that play the two cells of each comparison,
    at the main level of every setting, as the tables read them."""
    line = run_lines.run_line
    agents = [(line(design.AGENTS_PRESET, row), line(design.AGENTS_PRESET, ref)) for _, row, ref in design.HYPOTHESES]
    return {
        "agents": agents,
        "h0": agents,
        "h1_top": [(line(design.H1_TOP_PRESET, row), line(design.H1_TOP_PRESET, ref)) for _, row, ref in design.H1_TOP],
        "h2_prime": [(line(design.TRAINING_PRESET, row), line(design.TRAINING_PRESET, ref)) for _, row, ref in design.H2_PRIME],
        "full_environment": [(line(design.TRAINING_PRESET, row), line(design.AGENTS_PRESET, row)) for _, row in design.FULL_ENVIRONMENT],
        "layers": [(line(upper, row), line(lower, row)) for lower, upper in zip(design.LADDER, design.LADDER[1:]) for row in design.LADDER_ROWS],
        "parallel_time": [(line(design.PARALLEL_PRESET, design.PARALLEL_ROW),
                           line(design.PARALLEL_PRESET, design.PARALLEL_ROW, next(iter(SEQUENTIAL.values()))))],
    }


def _training_lines(row, variant):
    """The lines of the part training that play one variant of a row with the MCP, one per fold for a trained variant."""
    if variant == training.AS_RELEASED:
        return [run_lines.run_line(design.TRAINING_PRESET, row, f"--training-variant {variant}")]
    return [run_lines.run_line(design.TRAINING_PRESET, row, f"--training-variant {variant} --training-fold {fold}") for fold in training.held_out_folds.fold_ids()]


def test_every_cell_of_a_registered_comparison_plays_on_the_server_of_its_family():
    """Every family of the registered file but training compares cells that lines of grid_agents.txt play, all of whose parts play on a
    server without adapters, and the training family compares variants that grid_training_evaluation.txt alone plays, beside the
    adapters, so the two cells of a comparison never come from two servers. A family that the map of this test lacks fails it."""
    cells = _registered_cells()
    assert set(registered.load()["families"]) == set(cells) | {"training"}
    assert run_lines.ADAPTER_PARTS == ("training",)
    assert [line for _, line in run_lines.part("training")] == _file_lines("grid_training_evaluation.txt")
    plain, adapters = set(_file_lines("grid_agents.txt")), set(_file_lines("grid_training_evaluation.txt"))
    assert not [line for line in plain if "--training-variant" in line]
    for family, pairs in cells.items():
        for pair in pairs:
            assert set(pair) <= plain, (family, pair)
    for row in design.TRAINING_ROWS:
        for comparison in design.TRAINING_COMPARISONS:
            for variant in comparison[:2]:
                lines = _training_lines(row, variant)
                assert lines and set(lines) <= adapters and not set(lines) & plain, (row, variant)


def test_the_part_environment_plays_the_rows_with_the_mcp_on_the_full_environment_after_every_other_line():
    """The part environment holds the three rows with the MCP on all six big layers as released, the lines that the part training
    plays without training beside the adapters, and closes grid_agents.txt, so every earlier line keeps its place."""
    environment = [line for _, line in run_lines.part("environment")]
    assert environment == [run_lines.run_line(design.TRAINING_PRESET, row) for row in design.MCP_ROWS]
    assert _file_lines("grid_agents.txt")[-len(environment):] == environment
    assert run_lines.replicates("environment") == run_lines.replicates("main")
    released = [line for _, line in run_lines.part("training") if f"--training-variant {training.AS_RELEASED}" in line]
    assert [line.replace(f" --training-variant {training.AS_RELEASED}", "") for line in released] == environment


def test_the_page_serves_the_adapters_for_the_parts_that_need_them_alone():
    """A command line of the page that starts a server with SERVE_ADAPTERS=1 names a part of grid_lines.ADAPTER_PARTS, and every such
    part has one, so every other part plays on a server without adapters."""
    commands = [line for line in README.splitlines() if re.match(r"(E|P|T)=|bash ", line)]
    serving = [line for line in commands if "SERVE_ADAPTERS=1" in line]
    assert serving and all(any(f"GRID_PART={part}" in line for part in run_lines.ADAPTER_PARTS) for line in serving)
    assert all(any(f"GRID_PART={part}" in line for line in serving) for part in run_lines.ADAPTER_PARTS)


def test_run_json_records_the_lora_adapters_of_the_endpoint_file_of_its_endpoint(tmp_path, monkeypatch):
    """run.json lora_adapters is the line of the endpoint file that names the run's endpoint, and 0 for another endpoint, for no
    endpoint file and for an agent without a model."""
    path = tmp_path / "endpoint.txt"
    url = "http://127.0.0.1:30001/v1"
    path.write_text(f"{url}\njob 7 node n model m served_name m\nsglang 0.5.19\nlora_adapters 6\n", encoding="utf-8")
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(path))
    assert run_episodes.lora_adapters_of(SimpleNamespace(agent="causal", base_url=url)) == 6
    assert run_episodes.lora_adapters_of(SimpleNamespace(agent="causal", base_url="http://127.0.0.1:30002/v1")) == 0
    assert run_episodes.lora_adapters_of(SimpleNamespace(agent="oracle", base_url=url)) == 0
    monkeypatch.delenv("HYD_ENDPOINT_FILE")
    assert run_episodes.lora_adapters_of(SimpleNamespace(agent="causal", base_url=url)) == 0


def test_a_line_played_on_two_serving_setups_is_two_conditions():
    base = {"agent": "causal", "agent_label": f"causal_{MODEL}", "model": MODEL, "config_name": design.TRAINING_PRESET}
    assert pooling_rules.condition_of({**base, "lora_adapters": 0}) == pooling_rules.condition_of(base)
    assert pooling_rules.condition_of({**base, "lora_adapters": 6}) != pooling_rules.condition_of(base)


def test_a_finished_run_covers_a_line_only_on_the_serving_setup_of_the_job_that_asks(tmp_path, monkeypatch):
    """Inside a serving job a pair of a line and a replicate counts as done only for a run on the setup of that job, so the lines without
    training of the part training never count as runs of the part environment, and outside a job the setup checks nothing."""
    for name in ("HYD_ENDPOINT_FILE", "ENDPOINT_FILE"):
        monkeypatch.delenv(name, raising=False)
    line = run_lines.run_line(design.TRAINING_PRESET, design.CD_MCP) + " --replicate 1"
    assert "lora_adapters" not in pending_lines.expected(line)
    fields = {}
    for adapters in (0, 6):
        path = tmp_path / f"endpoint_{adapters}.txt"
        path.write_text(f"http://127.0.0.1:30001/v1\nlora_adapters {adapters}\n", encoding="utf-8")
        monkeypatch.setenv("ENDPOINT_FILE", str(path))
        fields[adapters] = pending_lines.expected(line)
        assert fields[adapters]["lora_adapters"] == adapters
    beside = dict(fields[6])
    assert pending_lines.matches(beside, fields[6], {}) and not pending_lines.matches(beside, fields[0], {})
    before = {key: value for key, value in fields[0].items() if key != "lora_adapters"}  # a run.json written before the key
    assert pending_lines.matches(before, fields[0], {}) and not pending_lines.matches(before, fields[6], {})


def test_the_training_tables_read_each_row_on_one_serving_setup():
    """A row of the training tables with a run beside the adapters reads the runs beside the adapters alone, and a row without one, the
    baseline played by the top rung of the ladder, reads the server without them."""
    def episode(server, adapters, seed):
        return {"model": MODEL, "surface": "aob", "agent": f"causal_{MODEL}", "diagnostics_server": server, "lora_adapters": adapters, "seed": seed}
    mixed = [episode(True, adapters, seed) for adapters in (0, 2) for seed in SEEDS]
    baseline = [episode(False, 0, seed) for seed in SEEDS]
    kept = aggregate.one_setup(mixed + baseline)
    assert sorted((e["diagnostics_server"], e["lora_adapters"]) for e in kept) == [(False, 0)] * len(SEEDS) + [(True, 2)] * len(SEEDS)


def test_the_agents_table_and_h2_prime_read_the_full_environment_on_the_server_without_adapters(tmp_path):
    """The CausalDiagnosis agent with the MCP on all six big layers as released, played on the server of the agents grid, where it
    passes, and again beside the adapters, where it fails: AGENTS holds both as rows of their own, the agents table reads the first,
    and H2' pairs it with the agent without the MCP of the top rung on the same server, five pairs and not ten."""
    runs = tmp_path / "runs"
    for name, row, adapters, passes in (("env_cd_mcp", design.CD_MCP, 0, True), ("train_cd_mcp", design.CD_MCP, 2, False),
                                        ("top_cd", design.CD, 0, False)):
        _run(runs, name, [_episode(s, row, design.TRAINING_PRESET, block=COMBINATIONS[s], **{"pass": passes}) for s in SEEDS],
             _meta(design.TRAINING_PRESET, lora_adapters=adapters))
    tables, page, _ = _aggregated(tmp_path, runs)
    rows = [r for r in tables["AGENTS"] if r["agent"] == f"causal_{MODEL}" and r["config"] == design.TRAINING_PRESET
            and str(r["diagnostics_server"]) == "True"]
    assert sorted(str(r["lora_adapters"]) for r in rows) == ["0", "2"]
    pairs = [r for r in tables["H2PRIME_pairs"] if r["agent"] == f"causal_{MODEL}"]
    assert pairs and {int(r["pairs"]) for r in pairs} == {len(SEEDS)} and {float(r["pass_diff"]) for r in pairs} == {1.0}
    name = next(name for prefix, with_mcp, name in design.AGENT_ROWS if prefix == "causal" and not with_mcp)
    agents = _lines(page, "agents")
    group = agents.index(next(cells for cells in agents if cells[0] == name))
    environment = next(cells for cells in agents[group:] if cells[0] == design.AGENT_LINES[-1][0])
    assert environment[-2].startswith("100.0")
