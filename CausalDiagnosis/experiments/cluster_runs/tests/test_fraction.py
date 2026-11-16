"""The share of the test set that a grid plays: two kinds of value, one default of a whole
replicate, the cut look on a
seeded prefix of the start combinations, the units and the share of the launchers, the condition of a cut run, and one test through
the path that ships with its control."""
from CausalDiagnosis.utils.paths import CLUSTER_JOBS, PACKAGE
import json
import math
import random
import re
import subprocess
from fractions import Fraction
from types import SimpleNamespace

import pytest

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines
from CausalDiagnosis.evaluation.result_tables import pooling_rules
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.datasets.splits.development_split import coverage_order, faulty_levels
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states

HPC = CLUSTER_JOBS
RUNNER = (CLUSTER_JOBS / "3_run" / "run_grid.sbatch").read_text(encoding="utf-8")


# ----- the rule -----

@pytest.mark.parametrize("text,whole", [("0.2", 1), ("0.4", 2), ("0.6", 3), ("0.8", 4), ("1", 5), ("1.0", 5), ("0.40", 2)])
def test_a_whole_share_names_its_replicates_by_exact_arithmetic(text, whole):
    assert start_states.fraction_plan(text) == ("whole", whole)


@pytest.mark.parametrize("text", ["0.05", "0.1", "0.15", "0.01"])
def test_a_share_below_one_replicate_is_a_cut_look(text):
    assert start_states.fraction_plan(text) == ("cut", Fraction(text))


@pytest.mark.parametrize("text,named", [("0.3", ("0.2", "0.4")), ("0.25", ("0.2", "0.4")), ("0.9", ("0.8", "1")), ("0.55", ("0.4", "0.6"))])
def test_a_share_inside_a_later_replicate_is_refused_with_the_nearest_whole_shares(text, named):
    with pytest.raises(ValueError) as refused:
        start_states.fraction_plan(text)
    assert all(f" {share}" in str(refused.value) for share in named) and "unequally" in str(refused.value)


@pytest.mark.parametrize("text,words", [("1/47", "ratio"), ("1/20", "ratio"), ("0.0212765957", "exact written form")])
def test_a_share_whose_written_form_is_not_exact_is_refused(text, words):
    """The units carry the written form of fraction_text and the driver cuts by it, so a share whose written form is not the
    value exactly would play another prefix in the driver than the share names (n/235 for n = 5 gives 6 combinations)."""
    with pytest.raises(ValueError) as refused:
        start_states.fraction_plan(text)
    assert words in str(refused.value)
    assert start_states.fraction_plan("0.05") == ("cut", Fraction(1, 20))  # the control: a short decimal stands


@pytest.mark.parametrize("text", ["0", "-0.1", "1.5", "x", ""])
def test_a_share_outside_zero_to_one_or_no_number_is_refused(text):
    with pytest.raises(ValueError):
        start_states.fraction_plan(text)


def test_the_default_share_stands_once_and_is_one_whole_replicate():
    assert start_states.DEFAULT_FRACTION == "0.2" and start_states.fraction_plan(start_states.DEFAULT_FRACTION) == ("whole", 1)
    assert start_states.fraction_text(Fraction(1, 20)) == "0.05" and start_states.fraction_value("0.05") == 0.05
    shell = [p for p in HPC.iterdir() if p.suffix in (".sh", ".sbatch")]
    restated = [p.name for p in shell if any(re.search(rf"(?<![0-9.]){re.escape(v)}(?![0-9])", p.read_text(encoding="utf-8"))
                                              for v in (start_states.DEFAULT_FRACTION, "0.05"))]
    assert restated == [], f"a script restates the default share or the former one: {restated}"


def _env(combos, components=("a", "b", "c")):
    healthy = {c: 0 for c in components}
    seed_of = {seed: combo for seed, combo in enumerate(combos)}
    return SimpleNamespace(candidates=list(combos), config=SimpleNamespace(components=list(components), healthy=healthy),
                           start_of=lambda seed: (seed_of[seed], 0, 0))


@pytest.mark.parametrize("count", [1, 7, 23, 47])
def test_a_cut_look_keeps_a_nested_prefix_of_the_coverage_order_of_its_size(count):
    rng = random.Random(count)
    combos = sorted({tuple(rng.randrange(4) for _ in range(3)) for _ in range(400)})[:count]
    assert len(combos) == count, "the draw holds enough distinct combinations"
    env = _env(combos)
    kept = {share: start_states.cut_combinations(env, share) for share in ("0.01", "0.05", "0.1", "0.15")}
    for share, combinations in kept.items():
        assert len(combinations) == math.ceil(Fraction(share) * start_states.REPLICATES * count) <= count, share
    assert kept["0.01"] <= kept["0.05"] <= kept["0.1"] <= kept["0.15"]
    covering, rest = coverage_order(combos, lambda c: faulty_levels(c, [0, 0, 0]), 42)
    assert (covering + rest)[:len(kept["0.05"])] == [c for c in covering + rest if c in kept["0.05"]][:len(kept["0.05"])]
    # the seeds of replicate 1 keep their order and drop the combinations outside the prefix
    seeds = list(range(count))
    assert start_states.cut_seeds(env, seeds, "0.05") == [s for s in seeds if combos[s] in kept["0.05"]]


def _capped_before(starts, cap, healthy, seed):
    """capped_starts as the tree held it before the coverage order moved into development_split.py, the control of the move."""
    order = list(starts)
    random.Random(seed).shuffle(order)
    taken, covered = [], set()
    for start in order:
        levels = faulty_levels(start, healthy)
        if levels - covered:
            taken.append(start)
            covered |= levels
    if len(taken) > cap:
        raise SystemExit("below")
    for start in order:
        if len(taken) == cap:
            break
        if start not in taken:
            taken.append(start)
    kept = set(taken)
    return [start for start in starts if start in kept]


def test_the_capped_development_starts_are_those_of_the_tree_before_the_shared_coverage_order():
    rng = random.Random(7)
    for trial in range(40):
        starts = sorted({tuple(rng.randrange(4) for _ in range(4)) for _ in range(30)})
        healthy = [0, 0, 0, 0]
        for cap in range(1, len(starts) + 1):
            try:
                before = _capped_before(starts, cap, healthy, trial)
            except SystemExit:
                with pytest.raises(SystemExit):
                    run_episodes.capped_starts(starts, cap, healthy, trial)
                continue
            assert run_episodes.capped_starts(starts, cap, healthy, trial) == before, (trial, cap)


def test_a_fold_line_of_a_cut_look_is_empty_only_when_no_combination_of_its_fold_is_in_the_prefix():
    combos = [(a, b, 0) for a in range(3) for b in range(3)]
    env = _env(combos)
    kept = start_states.cut_combinations(env, "0.05")
    inside, outside = sorted(kept)[0], next(c for c in combos if c not in kept)
    split = {c: (1 if c == inside else 2 if c == outside else 3, ()) for c in combos}
    assert not start_states.fold_line_empty(env, split, 1, "0.05")
    assert start_states.fold_line_empty(env, {**split, inside: (3, ())}, 1, "0.05")  # the control: the fold loses its one cell
    assert not any(start_states.fold_line_empty(env, split, fold, "0.2") for fold in (1, 2, 3)), "a whole share gives every fold its cells"


def test_the_wiring_agents_stand_once_beside_the_replicates_and_play_the_cut_look_on_replicate_1():
    assert pending_lines.WIRING_AGENTS is start_states.WIRING_AGENTS
    assert "WIRING_AGENTS = " not in (PACKAGE / "experiments" / "cluster_runs" / "pending_grid_lines.py").read_text(encoding="utf-8")
    line = "--config real_time --servers aob --agent scripted --no-diagnostics"
    assert pending_lines.units([line], replicates=(1,), option="--fraction 0.05") == [f"{line} --replicate 1 --fraction 0.05"]


# ----- the driver -----

@pytest.mark.parametrize("argv,words", [
    (["--config", "static", "--agent", "do_nothing", "--fraction", "0.05"], "needs --starts sampled"),
    (["--config", "static", "--agent", "do_nothing", "--starts", "sampled", "--fraction", "0.4"], "2 whole replicates"),
    (["--config", "static", "--agent", "do_nothing", "--starts", "sampled", "--fraction", "0.3"], "cuts inside replicate 2"),
    (["--config", "static", "--agent", "do_nothing", "--starts", "sampled", "--replicate", "2", "--fraction", "0.05"], "replicate 1 alone"),
])
def test_the_driver_refuses_a_share_that_plays_no_cut_look(tmp_path, argv, words):
    with pytest.raises(SystemExit) as refused:
        run_episodes.main([*argv, "--out", str(tmp_path / "runs")])
    assert words in str(refused.value)
    assert not (tmp_path / "runs").exists()


def test_the_driver_default_plays_the_whole_replicate_and_records_no_share():
    assert run_episodes.build_parser().get_default("fraction") is None


# ----- the units -----

def test_the_play_plan_gives_the_replicates_and_the_option_of_one_rule():
    assert pending_lines.play_plan() == ((1,), "") and pending_lines.play_plan("0.05") == ((1,), "--fraction 0.05")
    assert pending_lines.play_plan("0.4") == ((1, 2), "") and pending_lines.play_plan("1") == (pending_lines.REPLICATES, "")
    assert pending_lines.play_plan(None, [1, 3]) == ((1, 3), "")
    for both, words in ((("0.2", [1]), "set one"), (("0.3", None), "0.2 or 0.4")):
        with pytest.raises(SystemExit) as refused:
            pending_lines.play_plan(*both)
        assert words in str(refused.value)


def test_the_play_plan_refuses_a_replicate_outside_the_test_set():
    with pytest.raises(SystemExit) as refused:
        pending_lines.play_plan(None, [1, 6])
    assert "not 6" in str(refused.value)


def test_a_fold_line_of_a_cut_look_whose_fold_gets_no_cell_is_not_pending(tmp_path, monkeypatch):
    line = "--config real_time --servers aob --store couchdb --agent causal --training-variant ft --training-fold 2"
    asked = []
    monkeypatch.setattr(pending_lines, "fold_line_empty", lambda *args: asked.append(args) or args[1] == 2)
    unit = f"{line} --replicate 1 --fraction 0.05"
    assert pending_lines.empty_cut(unit, "sampled") and asked[0][:3] == ("real_time", 2, "0.05")
    assert not pending_lines.empty_cut(unit.replace("--training-fold 2", "--training-fold 1"), "sampled")  # the control
    assert not pending_lines.empty_cut(f"{line} --replicate 1", "sampled"), "a whole replicate gives every fold its cells"
    assert pending_lines.pending([line], [tmp_path], {}, "sampled", (1,), None, "--fraction 0.05") == []


def test_an_empty_fold_line_of_a_cut_look_ends_with_nothing_to_play_and_no_run_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_episodes.planned_episodes, "chosen_seeds", lambda *args: [])
    with pytest.raises(SystemExit) as ended:
        run_episodes.main(["--config", "static", "--agent", "do_nothing", "--starts", "sampled", "--fraction", "0.05",
                           "--out", str(tmp_path / "runs")])
    assert ended.value.code == 0 and "NOTHING_TO_PLAY" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_a_cut_line_carries_the_share_and_never_matches_a_whole_run():
    line = "--config real_time --servers aob --store couchdb --agent causal"
    assert pending_lines.units([line], replicates=(1,), option="--fraction 0.05") == [f"{line} --replicate 1 --fraction 0.05"]
    assert pending_lines.units([f"{line} --replicate 3"], replicates=(1,), option="--fraction 0.05") == [f"{line} --replicate 3"]
    cut, whole = pending_lines.expected(f"{line} --replicate 1 --fraction 0.05"), pending_lines.expected(f"{line} --replicate 1")
    assert cut["fraction"] == 0.05 and whole["fraction"] is None
    assert not pending_lines.matches({**whole, "ended": "t"}, cut, {}) and not pending_lines.matches({**cut, "ended": "t"}, whole, {})


def test_the_condition_of_a_whole_run_reads_as_before_the_share_and_a_cut_run_stands_apart():
    meta = {"agent_label": "causal_m", "config_name": "real_time", "replicate": 1}
    before = json.dumps({key: {**pooling_rules.ABSENT, **meta}.get(key) for key in pooling_rules.CONDITION_KEYS if key != "fraction"},
                        sort_keys=True, default=str)
    assert pooling_rules.condition_of(meta) == before
    assert pooling_rules.condition_of({**meta, "fraction": None}) == before
    assert pooling_rules.condition_of({**meta, "fraction": 0.05}) != before and '"fraction": 0.05' in pooling_rules.condition_of({**meta, "fraction": 0.05})
    assert "fraction" in pooling_rules.ROW_FIELDS and pooling_rules.ROW_FIELDS[-1] == "fraction"


def test_the_episode_set_of_a_whole_run_reads_as_before_the_share():
    meta = {"starts": "sampled", "master_seed": 42, "seed_pool": 8, "replicate": 2}
    before = {k: meta.get(k) for k in ("starts", "master_seed", "seed_pool", "replicate")}
    assert pooling_rules.ABSENT.get("fraction") is None and pending_lines.ABSENT["fraction"] is None
    source = (PACKAGE / "evaluation" / "result_tables" / "results_index" / "index_file.py").read_text(encoding="utf-8")
    assert '**({"fraction": meta["fraction"]} if meta.get("fraction") is not None else {})' in source
    assert {**before} == before  # the episode set of inputs() adds the key for a cut run alone


def test_a_rebuild_refuses_an_index_that_names_no_share(tmp_path):
    from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results

    with pytest.raises(SystemExit) as refused:
        verify_results.rebuild({"runs_root": "runs", "inputs": [], "parameters": {"resamples": 1, "alpha": 0.05, "seed": 0, "cluster": None}},
                               tmp_path / "results", {}, tmp_path)
    assert "names no share" in str(refused.value)


def test_every_call_of_the_read_names_its_share():
    import ast

    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "aggregate" \
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "aggregate":
                assert any(k.arg == "fraction" for k in node.keywords), f"{path.name}:{node.lineno} reads without naming its share"


def _bash(*parts, env=None):
    return subprocess.run(["bash", "-c", "\n".join(parts)], capture_output=True, text=True, env=env, timeout=60)


def test_the_grid_runner_takes_its_replicates_and_the_option_from_the_share_helper(tmp_path):
    function = re.search(r"^replicate_option\(\) \{.*\}$", RUNNER, re.M).group(0)
    done = _bash(function, 'FRACTION_OPTION="--fraction 0.05"', 'replicate_option 1; replicate_option none; FRACTION_OPTION=', "replicate_option 2")
    assert done.stdout.splitlines() == ["--replicate 1 --fraction 0.05", "--replicate 2"], done.stderr
    grid_lines = PACKAGE / "experiments" / "cluster_runs" / "grid_job_lists.py"
    for share, replicates, out in (("", "", "1\n\n"), ("0.05", "", "1\n--fraction 0.05\n"), ("0.4", "", "1 2\n\n"),
                                   ("", "1 3", "1 3\n\n")):
        done = subprocess.run(["python3", str(grid_lines), "share", share, replicates], capture_output=True, text=True, timeout=60)
        assert done.returncode == 0 and done.stdout == out, (share, replicates, done.stderr)
    done = subprocess.run(["python3", str(grid_lines), "share", "0.2", "1"], capture_output=True, text=True, timeout=60)
    assert done.returncode != 0 and "set one" in done.stderr


# ----- one test through the path that ships, with its control -----

LINES = ("--config static --agent do_nothing", "--config static --agent oracle")


def _play(root, fraction):
    """Plays the pending units of LINES at a share as run_lines.sh and run_grid.sbatch do: pending_grid_lines.py names them, the driver
    plays each into the root. Returns the units it played."""
    replicates, option = pending_lines.play_plan(fraction)
    played = pending_lines.pending(list(LINES), [root], {}, "sampled", replicates, None, option)
    for unit in played:
        run_episodes.main([*unit.split(), "--starts", "sampled", "--out", str(root)])
    return played


def _read(root, out, fraction):
    tables = dict((table.id, rows) for table, rows in aggregate.aggregate(root, out, fraction=fraction, pages=False))
    return tables["RUNS"], [v for rows in tables.values() for row in rows for v in row.values()]


def test_a_grid_plays_a_cut_look_then_the_default_whole_replicate_then_the_raise_reuses_it(tmp_path):
    root = tmp_path / "runs"
    first = _play(root, "0.05")
    assert first == [f"{line} --replicate 1 --fraction 0.05" for line in LINES]
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/run.json"))]
    assert len(metas) == 2 and all(m["fraction"] == 0.05 and m["fraction_rule"] == start_states.FRACTION_RULE for m in metas)
    rows, cells = _read(root, tmp_path / "cut", "0.05")
    assert len(rows) == 2 and all(row["pass_low"] == "" and row["pass_high"] == "" for row in rows), "a cut look prints estimates alone"
    assert any("partial" in str(cell) for cell in cells), "a read below the share of record is marked partial"
    # the default, a whole replicate: the cut runs stand for no unit of it
    assert _play(root, None) == [f"{line} --replicate 1" for line in LINES]
    rows, cells = _read(root, tmp_path / "one", start_states.DEFAULT_FRACTION)
    assert any("partial" in str(cell) for cell in cells), "the default read lies below the share of record"
    assert sorted(str(row["replicate"]) for row in rows) == ["1", "1"]
    # the raise plays the missing replicate alone, and the read at 0.4 holds both replicates of each line
    kept = sorted(p.parent.name for p in root.glob("*/run.json"))
    assert _play(root, "0.4") == [f"{line} --replicate 2" for line in LINES]
    assert set(kept) <= {p.parent.name for p in root.glob("*/run.json")}
    rows, cells = _read(root, tmp_path / "two", "0.4")
    assert sorted(str(row["replicate"]) for row in rows) == ["1 2", "1 2"]
    assert not any("incomplete" in str(cell) for cell in cells)
    # the control: a line whose replicate 2 is removed reads incomplete at 0.4
    second = next(p.parent for p in root.glob("*/run.json")
                  if json.loads(p.read_text(encoding="utf-8")).get("replicate") == 2 and "oracle" in json.loads(p.read_text(encoding="utf-8"))["agent"])
    (second / "run.json").rename(second / "run.json.removed")
    _rows, cells = _read(root, tmp_path / "control", "0.4")
    assert any("incomplete" in str(cell) for cell in cells), "a line that lacks a replicate of the share reads incomplete"
