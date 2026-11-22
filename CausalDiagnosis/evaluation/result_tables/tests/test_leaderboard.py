"""CausalDiagnosis/evaluation/result_tables/leaderboard.py on synthetic run folders made with the helpers of test_tables_from_runs.py and
aggregated by tables_from_runs/, so that no recorded value enters: the six rows of design.AGENT_ROWS on layer 0 in their order
with the oracle policy and the declaring policy under them, the still running and judge pending cells, the pass rate with
its interval clustered by start, the cutoff and the episode set, the environment table on the presets of design.LADDER with
the three agents with the CausalDiagnosis MCP, the ablation lines with the presets they follow, the runs that the page
leaves out with their reason, and the loud failure on a label, switch or preset that no row or ablation maps."""
from CausalDiagnosis.utils.paths import CLUSTER_JOBS, RESULTS, preset_files
import json

import pytest

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.experiments.episode_runners import run_configuration
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.evaluation.result_tables import leaderboard
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines
from CausalDiagnosis.experiments.run_verification import rescore_under_verifier_scope as rescore_scope
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.evaluation.metrics import table_statistics as stats
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, read_rows
from CausalDiagnosis.experiments.episode_runners.run_configuration import CONFIGS, load_config
from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_RANKING_MODEL
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import METERED_SERVERS, RIG_SERVER
from CausalDiagnosis.utils.csv_rows import write_rows
from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import _row, _run

# the model agents play the rows over AssetOpsBench's servers and the task server, a reference policy over the
# rig's action server and without a model
DEVELOPMENT = {"starts": "combinations", "started": "2026-11-02T13:00:00", "config_name": design.AGENTS_PRESET, "servers": list(METERED_SERVERS)}
REFERENCE = {**DEVELOPMENT, "model": None, "servers": [RIG_SERVER]}
CUTOFF = "plan_execute=2026-11-02T12:00:00"
# the names of the six rows: each agent without and then with the CausalDiagnosis MCP
LABELS = ["Plan-and-Execute", "Plan-and-Execute + CausalDiagnosis MCP", "Agents-as-Tools",
          "Agents-as-Tools + CausalDiagnosis MCP", "CausalDiagnosis agent", "CausalDiagnosis agent + CausalDiagnosis MCP"]
PE, PE_MCP, AAT, AAT_MCP, CD, CD_MCP = LABELS
LADDER_AGENTS = [PE_MCP, AAT_MCP, CD_MCP]
BASE_POINTER = "the rows with the CausalDiagnosis MCP of the main table above"
# the supporting ablations in the order of the leaderboard: the scoring method, the parts of the environment and of the
# MCP, the small switches of layers 2, 3 and 5 under the ids of their layers, and the switches of the CausalDiagnosis agent
PAPER_IDS = ["Sc3", "En4", "En5", "En6", "En7", "En9", "L1", "L2", "L3", "L5", *[f"Ag{i}" for i in (1, 2, 3, 7, 9, 10)]]
ARM_PRESETS = {"deterioration_thinking_off": ("L2", "the thinking time off"),
               "deterioration_thinking_price_double": ("L2", "the thinking time at twice its price"),
               "deterioration_thinking_price_half": ("L2", "the thinking time at half its price"),
               "deterioration_durations_off": ("L3", "the declared action durations off"),
               "deterioration_mode_none": ("L5", "the deterioration off"), "scheduled": ("L5", "the fixed schedule of level drops against the random hazard rates"),
               "deterioration_failure_off": ("L5", "the failure rule off"),
               "deterioration_misdiagnosis_budget": ("L1", "the misdiagnosis price at the budget")}
UNREPORTED_REASON = "which no table of the page reads since no row and no supporting ablation plays this preset"


def _on(preset, seed, **fields):
    """An episode row on a preset in the preset's own clock mode (configs/<preset>.json) with the driver's ranking model."""
    return _row(seed, **{"config": preset, "clock_mode": load_config(preset)[1].clock_mode, "ranking_model": DEFAULT_RANKING_MODEL, **fields})


def _main_row(seed, **fields):
    """An episode row of the main table: layer 0, the base preset, in its declared clock mode."""
    return _on(design.AGENTS_PRESET, seed, **fields)


def _running(root, preset, label, variant=None, **fields):
    """A run folder in progress: the first row of its scores.csv, no run.json, the start time in its name and, for a trained agent,
    the ledger that the driver writes before the first episode with the training variant in its configuration."""
    folder = root / f"{preset}_{label}_20261102-130000"
    folder.mkdir(parents=True)
    write_rows(folder / "scores.csv", [_on(preset, 0, agent=label, **fields)], COLUMNS)
    if variant:
        (folder / "ledger.json").write_text(json.dumps({"status": "running", "configuration": {"training_variant": variant}}), encoding="utf-8")
    return folder


def _as_run(row, name="x", **fields):
    """A row of RUNS.csv for one episode row, the settings as aggregate.table_runs writes them, every cell a string."""
    return {k: str(v) for k, v in {**row, "run": name, "mode": "gym", "abstention_rule": "decision", "training_variant": design.AS_RELEASED,
                                   "started": "2026-11-02T13:00:00", **fields}.items()}


# the run of an ablation plays on the preset of its place (design.ABLATION_PLACES), plan length one on the place of Ag3
LEN1_PLACE = design.ABLATION_PLACES["Ag3"][0]


def _runs(root):
    """The six rows of the main table and what the page reads around them: Plan-Execute without the MCP has no run,
    Plan-Execute with it one run before the cutoff and one after, Agent-As-Tool without the MCP misses one judge verdict,
    then the oracle, plan length one of Ag3 on its place, a run of Agent-As-Tool with the MCP on the timed preset, which no line
    reads, and a run with the candidate troubleshooting plans off that has no run.json yet."""
    _run(root, "pe_mcp_old", [_main_row(s, agent="plan_execute_m", **{"pass": True}) for s in range(4)], {**DEVELOPMENT, "started": "2026-11-02T11:00:00"})
    _run(root, "pe_mcp", [_main_row(s, agent="plan_execute_m", **{"pass": s == 0}) for s in range(4)], DEVELOPMENT)
    _run(root, "aat_bare", [_main_row(s, agent="agent_as_tool_m", diagnostics_server=False, judge_pass="" if s == 3 else False) for s in range(4)], DEVELOPMENT)
    _run(root, "aat_mcp", [_main_row(s, agent="agent_as_tool_m") for s in range(4)], DEVELOPMENT)
    _run(root, "causal_bare", [_main_row(s, agent="causal_m", diagnostics_server=False, **{"pass": s < 3}) for s in range(4)], DEVELOPMENT)
    _run(root, "causal_mcp", [_main_row(s, agent="causal_m") for s in range(4)], DEVELOPMENT)
    _run(root, "oracle", [_main_row(s, agent="oracle", model="") for s in range(4)], REFERENCE)
    _run(root, "len1", [_on(LEN1_PLACE, s, agent="causal_len1_m") for s in range(4)], {**DEVELOPMENT, "config_name": LEN1_PLACE})
    _run(root, "aat_timed", [_on("timed", s, agent="agent_as_tool_m") for s in range(4)], {**DEVELOPMENT, "config_name": "timed"})
    _running(root, design.AGENTS_PRESET, "causal_nobranch_m")
    return root


def _page(tmp_path, runs):
    out = tmp_path / "aggregate"
    aggregate.aggregate(runs, out, resamples=50, readme_path=tmp_path / "README.md", published_root=tmp_path / "published", fraction="1")
    leaderboard.main(["--aggregate", str(out), "--out", str(tmp_path / "LEADERBOARD.md"), "--since", CUTOFF, "--runs", str(runs)])
    return (tmp_path / "LEADERBOARD.md").read_text(encoding="utf-8")


def _index(tmp_path):
    """The leaderboard section of the INDEX.json that tables_from_runs/ wrote and leaderboard.py filled."""
    return json.loads((tmp_path / "INDEX.json").read_text(encoding="utf-8"))["leaderboard"]


def _left_out(tmp_path):
    return {entry["run"]: entry["reason"] for entry in _index(tmp_path)["left_out"]}


def _sections(text, model="m"):
    """(main table, environment table, ablation table) of one model's part of the page."""
    main, rest = text.split(f"## Main comparison, {model}\n")[1].split(f"## Environment table, {model}\n")
    environment, ablations = rest.split(f"## Ablations, {model}\n")
    return main, environment, ablations


def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _rows(section):
    """The cells of every line of the section's table below its header."""
    return [_cells(line) for line in section.splitlines() if line.startswith("| ")][1:]


def _lines(section, first):
    """The cells of every table line whose first cell is `first`."""
    return [_cells(line) for line in section.splitlines() if line.startswith(f"| {first} |")]


def test_the_main_table_has_the_six_rows_in_order_with_still_running_and_judge_pending(tmp_path):
    runs = _runs(tmp_path / "runs")
    _running(runs, design.AGENTS_PRESET, "plan_execute_m", diagnostics_server=False)
    _running(runs, design.AGENTS_PRESET, "agent_as_tool_m")  # a run in progress leaves a finished run of its row on the line
    text = _page(tmp_path, runs)
    assert text.startswith("# Leaderboard of the hydraulic rig gym") and "preliminary" in text and "Columbia's cluster" in text
    main, _, _ = _sections(text)
    assert [row[0] for row in _rows(main)] == LABELS
    pe, pe_mcp, aat, aat_mcp, cd, cd_mcp = (_lines(main, label)[0] for label in LABELS)
    assert pe[1:4] == ["m", "base", "combinations"] and pe[4:] == ["still running", "", "", "", "", ""]
    # the run before the cutoff, which passes every episode, does not count
    assert pe_mcp[4] == "4" and pe_mcp[5].startswith("0.250 [") and pe_mcp[6] == "1.000" and pe_mcp[9] == "`pe_mcp`"
    assert aat[6] == "judge pending" and aat[9] == "`aat_bare`"
    assert aat_mcp[6] == "1.000" and aat_mcp[9] == "`aat_mcp`"
    assert cd[5].startswith("0.750 [") and cd[9] == "`causal_bare`" and cd_mcp[9] == "`causal_mcp`"
    assert "in the run folder `oracle`" in main and "is the best of the five reference policies and not a row of this table." in main


def test_the_main_table_holds_the_six_rows_of_design_on_layer_0_with_the_causal_diagnosis_agent_without_the_mcp(tmp_path):
    """The main table plays layer 0, the base preset, with the six rows of design.AGENT_ROWS in their order and
    under their names, each agent without and then with the CausalDiagnosis MCP. The CausalDiagnosis agent without the MCP is
    a row of its own, filled by the agent's label with diagnostics_server off, and never by the run with the MCP."""
    assert design.AGENTS_PRESET == "base" and [name for _, _, name in design.AGENT_ROWS] == LABELS
    assert leaderboard.MAIN_ROWS is design.AGENT_ROWS
    for i, (prefix, mcp, _) in enumerate(design.AGENT_ROWS):
        assert leaderboard.classify(_as_run(_main_row(0, agent=f"{prefix}_m", diagnostics_server=mcp))) == leaderboard.Slot("main", i)
    runs = tmp_path / "runs"
    folders = [f"{prefix}_{'mcp' if mcp else 'bare'}" for prefix, mcp, _ in design.AGENT_ROWS]
    for folder, (prefix, mcp, _) in zip(folders, design.AGENT_ROWS):
        _run(runs, folder, [_main_row(s, agent=f"{prefix}_m", diagnostics_server=mcp) for s in range(4)], DEVELOPMENT)
    main, _, _ = _sections(_page(tmp_path, runs))
    rows = _rows(main)
    assert [row[0] for row in rows] == LABELS and {row[2] for row in rows} == {"base"}
    assert [row[9] for row in rows] == [f"`{folder}`" for folder in folders]


def test_a_line_of_five_replicate_runs_is_one_line_of_the_page_over_their_pooled_episodes(tmp_path):
    """A line of the test set is five runs, one per replicate, which tables_from_runs/ pools into one row
    of RUNS, so the line of the page counts the episodes of every replicate and names every folder, in replicate order."""
    runs = _runs(tmp_path / "runs")
    for r in range(1, 6):
        seeds = list(range(10 * r, 10 * r + 4))
        _run(runs, f"causal_r{r}", [_main_row(s, agent="causal_m", **{"pass": r == 1}) for s in seeds],
             {**DEVELOPMENT, "starts": "sampled", "master_seed": 42, "seed_pool": 8, "seeds": seeds, "replicate": r, "replicate_rule": "rule",
              "episodes_planned": 4, "started": f"2026-11-02T1{r}:00:00"})
    main, _, _ = _sections(_page(tmp_path, runs))
    causal, = _lines(main, CD_MCP)
    assert causal[3:5] == ["evaluation, master seed 42", "20"] and causal[5].startswith("0.200 [")
    assert causal[9] == " ".join(f"`causal_r{r}`" for r in range(1, 6))


def test_a_page_without_any_run_states_that_no_run_has_been_played_and_marks_every_line_not_run(tmp_path):
    """The result tables carry no run result until the test set is played, so the page written from no run
    folder says so, marks every line not run and holds no sentence on numbers or on runs in progress."""
    (tmp_path / "runs").mkdir()
    text = _page(tmp_path, tmp_path / "runs")
    assert "No run has been played on the current code, so the tables below hold no number." in text
    assert "until the test set is played" in text and "from 0 counted run folders" in text
    assert "preliminary" not in text and "They come from the runs" not in text and "still running" not in text
    main, environment, ablations = _sections(text, "no model yet")
    assert [row[0] for row in _rows(main)] == LABELS and {row[4] for row in _rows(main)} == {"not run"}
    assert main.count("On the same set it has not run.") == 2  # the oracle policy and the declaring policy
    assert {row[5] for row in _rows(environment)[1:]} == {"not run"}
    assert {row[8] for row in _rows(ablations)} == {"not run"}


def test_the_ablation_table_names_every_ablation_and_marks_the_runs_in_progress(tmp_path):
    _, _, ablations = _sections(_page(tmp_path, _runs(tmp_path / "runs")))
    ids = [row[0] for row in _rows(ablations)]
    assert list(dict.fromkeys(ids)) == [ablation for ablation, _, _ in leaderboard.ABLATIONS] == PAPER_IDS
    len1, = _lines(ablations, "Ag3")
    assert len1[2:4] == ["the plan length", "plan length 1" + leaderboard.on_preset(LEN1_PLACE)]
    assert len1[6] == LEN1_PLACE and len1[4] == CD_MCP and len1[13] == "`len1`"
    assert _lines(ablations, "Ag1")[0][8] == "still running" and _lines(ablations, "Ag2")[0][8] == "not run"
    # Ag7 has two arms and no run yet, so it shows one line without an arm
    assert _lines(ablations, "Ag7")[0][2:4] == ["the read the board advises", ""] and _lines(ablations, "Ag9")[0][8] == "not run"
    # the timed preset plays no row and no arm, so its run sits on no line
    assert "`aat_timed`" not in ablations


def test_the_declaring_policy_stands_under_the_main_table_with_its_success_and_reward_and_is_no_row(tmp_path):
    """By hand: two of four declarations are correct and earn 0.5, the others earn nothing, so the sentence reads a
    success of one half and a mean reward of a quarter, where the pass of the policy is zero on every start."""
    runs = _runs(tmp_path / "runs")
    _run(runs, "declare", [_main_row(s, agent="declare", model="", success=s < 2, reward=0.5 if s < 2 else 0.0, **{"pass": False}) for s in range(4)], REFERENCE)
    main, _, _ = _sections(_page(tmp_path, runs))
    assert "The declaring policy, which reads the cheapest sensor set once, cites that reading and declares insufficient evidence on every start," in main
    assert "it has a verifier success rate of 0.500 [" in main and "and a mean reward of 0.250 over 4 episodes, in the run folder `declare`." in main
    assert [row[0] for row in _rows(main)] == LABELS
    section = leaderboard.index_section(leaderboard.read_runs(tmp_path / "aggregate"), dict([CUTOFF.split("=", 1)]), [])
    assert next(line for line in section["lines"] if line["run"] == "declare")["line"] == "the declaring policy"


def test_a_run_of_the_declaring_policy_moves_no_table_to_its_set_and_reads_still_running_while_it_plays(tmp_path):
    """A reference decides no set of episodes, as the oracle does not, so a run of it on the evaluation set leaves the tables on the
    combinations set, and a run of it without a run.json reads still running in the sentence."""
    runs = _runs(tmp_path / "runs")
    _run(runs, "declare_eval", [_main_row(s, agent="declare", model="") for s in range(4)], {**REFERENCE, "starts": "sampled", "master_seed": 42})
    _running(runs, design.AGENTS_PRESET, "declare", model="")
    main, _, _ = _sections(_page(tmp_path, runs))
    assert _lines(main, CD_MCP)[0][3] == "combinations"
    assert "and is not a row of this table. On the same set it is still running." in main


def test_a_running_loop_of_gepa_is_no_run_in_progress_of_the_page(tmp_path):
    """A run folder of the online loop of GEPA marks its rows with their GEPA call and has no run.json while it runs, so the scan
    for runs in progress leaves it out instead of counting a loop's episode as a run of the CausalDiagnosis agent."""
    runs = tmp_path / "runs"
    for name, preset, fields in ((f"{design.TRAINING_PRESET}_gepa_online_verifier_m_20261102-130000", design.TRAINING_PRESET, {"gepa_call": 1}),
                                 (f"{design.AGENTS_PRESET}_causal_m_20261102-130000", design.AGENTS_PRESET, {})):
        (runs / name).mkdir(parents=True)
        write_rows(runs / name / "scores.csv", [_on(preset, 0, agent="causal_m", **fields)], COLUMNS)
    assert [run["run"] for run in leaderboard.runs_in_progress([runs])] == [f"{design.AGENTS_PRESET}_causal_m_20261102-130000"]


def test_the_tables_move_to_the_evaluation_set_once_a_run_played_it(tmp_path):
    runs = _runs(tmp_path / "runs")
    _run(runs, "causal_eval", [_main_row(s, agent="causal_m") for s in range(4)], {**DEVELOPMENT, "starts": "sampled", "master_seed": 42, "started": "2026-11-02T14:00:00"})
    main, environment, _ = _sections(_page(tmp_path, runs))
    cd_mcp, = _lines(main, CD_MCP)
    assert cd_mcp[3] == "evaluation, master seed 42" and cd_mcp[9] == "`causal_eval`"
    assert _lines(main, AAT_MCP)[0][4] == "not run"  # its only run played the combinations set and is finished, so nothing is in flight
    assert _rows(environment)[0][4] == "evaluation, master seed 42"  # the tables of a model share one set of episodes


def test_every_label_the_driver_writes_maps_and_any_other_stops_the_script(tmp_path):
    parser = run_episodes.build_parser()
    for agent in run_episodes.AGENTS:
        options = run_episodes.causal_options(parser.parse_args(["--agent", agent])) if agent == "causal" else None
        leaderboard.framework_of(run_episodes.agent_label_of(agent, "m", options), "m")
    # every switch of the causal agent writes a label suffix that one ablation of the Agent family names, while the label of the rule
    # only switch, which left the driver with its ablation, and the labels of the expectation checks and of replanning, whose rows the experiments no longer play while their
    # switches stay, stop the script like any unknown switch
    switches = [action.option_strings[0] for action in parser._actions if action.option_strings[0].startswith("--causal-") and action.nargs == 0]
    flags = [[flag] for flag in switches] + [["--causal-plan-length", "1"]]
    labels = [run_episodes.agent_label_of("causal", "m", run_episodes.causal_options(parser.parse_args(["--agent", "causal", *flag]))) for flag in flags]
    left = {"causal_noexp_m", "causal_noback_m"}
    assert left <= set(labels)
    assert sorted(leaderboard.framework_of(label, "m")[1][0] for label in labels if label not in left) == ["Ag1", "Ag10", "Ag2", "Ag3", "Ag7", "Ag7", "Ag9"]
    for label, problem in (("causal_rule_m", "switch 'rule'"), ("react_m", "unknown agent label"), ("causal_fast_m", "switch 'fast'"),
                           ("causal_noexp_m", "switch 'noexp'"), ("causal_noback_m", "switch 'noback'"),
                           ("causal_nobranch_single_m", "several switches")):
        with pytest.raises(ValueError, match=problem):
            leaderboard.framework_of(label, "m")
    runs = tmp_path / "runs"
    _run(runs, "react", [_main_row(s, agent="react_m") for s in range(2)], DEVELOPMENT)
    with pytest.raises(ValueError, match="unknown agent label 'react_m'"):
        _page(tmp_path, runs)
    _run(tmp_path / "other", "custom", [_main_row(s, agent="causal_m", config="my_preset") for s in range(2)], DEVELOPMENT)
    with pytest.raises(ValueError, match="preset 'my_preset'"):
        _page(tmp_path / "x", tmp_path / "other")


def test_the_presets_follow_the_design_and_the_small_switches_are_presets_of_their_own():
    """The main table plays layer 0 and the training the deterioration row. The small switches of
    layers 2, 3 and 5 are presets of all six big layers with one setting changed (design.ARM_PRESETS), the presets that no row and no arm
    plays any more are left out, and the ablations are the supporting ablations of the paper."""
    assert design.AGENTS_PRESET == "base" and design.TRAINING_PRESET == "deterioration"
    assert design.ARM_PRESETS == ARM_PRESETS
    assert set(design.UNREPORTED) == {"static", "timed", "static_zero_durations"}
    assert [ablation for ablation, _, _ in leaderboard.ABLATIONS] == PAPER_IDS
    assert set(design.ABLATION_PLACES) == set(PAPER_IDS)
    assert all(design.ABLATION_PLACES[ablation] == (design.TRAINING_PRESET, design.FRAMEWORK_ROWS) for ablation, _ in design.ARM_PRESETS.values())
    # every preset that a line of the page names is a preset of the configs folder, and a run of any other preset stops the script
    presets = {path.stem for path in preset_files(CONFIGS).values()}
    named = set(design.LADDER) | set(design.ARM_PRESETS) | set(design.OWN_TABLES) | set(design.UNREPORTED) | set(design.TOKEN_PRESETS)
    assert named <= presets
    for preset in sorted(presets - named):
        with pytest.raises(ValueError, match=f"plays the preset '{preset}', which no row or ablation of the leaderboard maps"):
            leaderboard.classify(_as_run(_on(preset, 0, agent="causal_m")))
    assert leaderboard.level("base", "clock_mode") == "declared" and leaderboard.level("real_time", "clock_mode") == "generation_time"
    assert leaderboard.level("deterioration_thinking_off", "clock_mode") == "declared"


def test_a_run_of_an_arm_preset_is_the_arm_of_its_small_switch_and_the_presets_no_arm_plays_are_left_out(tmp_path):
    """A run of a preset of design.ARM_PRESETS at every level of that preset is the arm of its small switch on the
    ablation table, named by the arm alone, for the three agents without the MCP, while a run on the static or the timed preset, which
    no row or arm plays any more, sits on no line and carries its reason in INDEX.json, a switch on static episodes included."""
    runs = _runs(tmp_path / "runs")
    _run(runs, "causal_thinking_off", [_on("deterioration_thinking_off", s, agent="causal_m", diagnostics_server=False) for s in range(4)],
         {**DEVELOPMENT, "config_name": "deterioration_thinking_off"})
    _run(runs, "pe_failure_off", [_on("deterioration_failure_off", s, agent="plan_execute_m", diagnostics_server=False) for s in range(4)],
         {**DEVELOPMENT, "config_name": "deterioration_failure_off"})
    _run(runs, "aat_price_double", [_on("deterioration_thinking_price_double", s, agent="agent_as_tool_m", diagnostics_server=False) for s in range(4)],
         {**DEVELOPMENT, "config_name": "deterioration_thinking_price_double"})
    _run(runs, "causal_static", [_on("static", s, agent="causal_m") for s in range(4)], {**DEVELOPMENT, "config_name": "static"})
    _run(runs, "nobranch_static", [_on("static", s, agent="causal_nobranch_m") for s in range(4)], {**DEVELOPMENT, "config_name": "static"})
    text = _page(tmp_path, runs)
    main, _, ablations = _sections(text)
    assert _lines(main, CD_MCP)[0][2] == "base" and _lines(main, CD_MCP)[0][9] == "`causal_mcp`"
    l2 = {line[3]: line for line in _lines(ablations, "L2")}
    assert set(l2) == {"the thinking time off", "the thinking time at twice its price"}  # a point of the time pressure curve
    assert l2["the thinking time off"][2:7] == ["the thinking time of layer 2", "the thinking time off", CD, "m", "deterioration_thinking_off"]
    assert l2["the thinking time off"][13] == "`causal_thinking_off`"
    assert l2["the thinking time at twice its price"][4:7] == [AAT, "m", "deterioration_thinking_price_double"]
    assert l2["the thinking time at twice its price"][13] == "`aat_price_double`"
    l5 = {line[3]: line for line in _lines(ablations, "L5")}
    assert set(l5) == {"the failure rule off"} and l5["the failure rule off"][4:7] == [PE, "m", "deterioration_failure_off"]
    assert l5["the failure rule off"][13] == "`pe_failure_off`"
    assert _lines(ablations, "L3")[0][8] == "not run"
    for name in ("causal_static", "nobranch_static", "aat_timed"):
        assert f"`{name}`" not in text
    left_out = _left_out(tmp_path)
    assert left_out["causal_static"] == left_out["nobranch_static"] == f"a run of the preset static, {UNREPORTED_REASON}"
    assert left_out["aat_timed"] == f"a run of the preset timed, {UNREPORTED_REASON}"


def test_a_clock_mode_the_preset_does_not_set_stops_the_script_and_the_old_clock_arms_are_left_out(tmp_path):
    """The thinking time off is the preset deterioration_thinking_off and no longer a clock mode of the driver, so a run
    in another clock mode than its preset's stops the script on any preset, an arm with the MCP is refused, and a run of
    static_zero_durations.json, an earlier clock arm, is left out with its reason, while real_time_declared.json is no preset
    any more."""
    runs = _runs(tmp_path / "runs")
    _run(runs, "causal_zero", [_on("static_zero_durations", s, agent="causal_m") for s in range(4)], {**DEVELOPMENT, "config_name": "static_zero_durations"})
    text = _page(tmp_path, runs)
    assert "`causal_zero`" not in text
    assert _left_out(tmp_path)["causal_zero"] == f"a run of the preset static_zero_durations, {UNREPORTED_REASON}"
    with pytest.raises(ValueError, match="plays the clock mode declared on the preset 'deterioration', while the thinking time off is a preset of its own"):
        leaderboard.classify(_as_run(_on("deterioration", 0, agent="causal_m", diagnostics_server=False, clock_mode="declared"), "declared"))
    with pytest.raises(ValueError, match="plays the clock mode instant on the preset 'base', while the thinking time off is a preset of its own"):
        leaderboard.classify(_as_run(_main_row(0, agent="plan_execute_m", clock_mode="instant"), "instant"))
    with pytest.raises(ValueError, match="plays L2 with CausalDiagnosis agent \\+ CausalDiagnosis MCP, a row that no supporting ablation gives it"):
        leaderboard.classify(_as_run(_on("deterioration_thinking_off", 0, agent="causal_m"), "mcp"))


def test_the_copies_that_rescore_scope_writes_are_the_arm_of_en4_and_their_runs_stay_the_main_rows(tmp_path):
    """En4 is the runs of the six rows scored again. rescore_under_verifier_scope.py writes beside each main run a copy whose rows and run.json
    carry the verifier scope all and keep the start of the run, so each copy of a counted run is the En4 line of its agent, the
    run it read stays the line of the main table, the copy of the run before the cutoff is left out like that run, and the
    pass of an En4 line needs the read gate, which these copies do not hold on their first two seeds."""
    runs = _runs(tmp_path / "runs")
    main_runs = ("pe_mcp_old", "pe_mcp", "aat_bare", "aat_mcp", "causal_bare", "causal_mcp")
    for name in main_runs:
        source, copy = runs / name, name + rescore_scope.SUFFIX
        rows = read_rows(source)
        meta = json.loads((source / "run.json").read_text(encoding="utf-8"))
        _run(runs, copy, [rescore_scope.scoped_row(row, {"read_gate": int(row["seed"]) >= 2}, copy) for row in rows], rescore_scope.scoped_meta(meta, source, len(rows)))
    main, _, ablations = _sections(_page(tmp_path, runs))
    assert [_lines(main, label)[0][9] for label in LABELS[1:]] == [f"`{name}`" for name in main_runs[1:]]
    en4 = _lines(ablations, "En4")
    assert [line[3:7] for line in en4] == [["verifier scope all", label, "m", "base"] for label in LABELS[1:]]
    assert [line[13] for line in en4] == [f"`{name}{rescore_scope.SUFFIX}`" for name in main_runs[1:]]
    assert en4[-1][9].startswith("0.500 [") and _lines(main, CD_MCP)[0][5].startswith("1.000 [")  # the same run with and without the gate
    left_out = _left_out(tmp_path)
    assert left_out["pe_mcp_old_scope_all"] == left_out["pe_mcp_old"] == "started 2026-11-02T11:00:00, before the cutoff 2026-11-02T12:00:00 of plan_execute"


def test_the_environment_table_lists_every_row_of_the_ladder_in_order_with_the_three_agents_with_the_mcp(tmp_path):
    runs = _runs(tmp_path / "runs")
    _run(runs, "pe_priced", [_on("priced", s, agent="plan_execute_m") for s in range(4)], {**DEVELOPMENT, "config_name": "priced"})
    _run(runs, "aat_real_time", [_on("real_time", s, agent="agent_as_tool_m") for s in range(4)], {**DEVELOPMENT, "config_name": "real_time"})
    _run(runs, "causal_deterioration", [_on("deterioration", s, agent="causal_m") for s in range(4)],
         {**DEVELOPMENT, "config_name": "deterioration"})
    _run(runs, "seq_full", [_on(design.TRAINING_PRESET, s, agent="causal_seq_m") for s in range(4)], {**DEVELOPMENT, "config_name": design.TRAINING_PRESET})
    _, environment, ablations = _sections(_page(tmp_path, runs))
    lines = _rows(environment)
    names = [name for _, name in leaderboard.ENVIRONMENT_ROWS]
    assert list(dict.fromkeys(line[0] for line in lines)) == names
    assert len(lines) == 1 + len(LADDER_AGENTS) * (len(names) - 1)  # layer 0 is one line that points at the main table
    by_line = {(line[0], line[1]): line for line in lines}
    assert by_line[("+ priced reads and writes", PE_MCP)][3] == "priced" and by_line[("+ priced reads and writes", PE_MCP)][10] == "`pe_priced`"
    assert by_line[("+ priced reads and writes", AAT_MCP)][5] == "not run"
    assert by_line[("+ real-time clock", AAT_MCP)][3] == "real_time" and by_line[("+ real-time clock", AAT_MCP)][10] == "`aat_real_time`"
    assert by_line[("+ deterioration and failure", CD_MCP)][10] == "`causal_deterioration`"
    ag9, = _lines(ablations, "Ag9")  # Ag9 plays on its ablation place, the full environment with all six big layers on, which its line names
    assert ag9[3:7] == [leaderboard.SWITCHES["seq"][1] + ", on the row + deterioration and failure", CD_MCP, "m", design.TRAINING_PRESET]
    assert ag9[13] == "`seq_full`"


def test_the_environment_table_plays_the_ladder_with_the_three_agents_with_the_mcp_and_points_layer_0_at_the_main_table(tmp_path):
    """One line per later row of design.LADDER and agent of design.LADDER_ROWS, in the order of the ladder and of
    the main table, and layer 0 as one line that points at the main table, whose rows with the MCP are its runs."""
    assert [name for _, _, name in design.LADDER_ROWS] == LADDER_AGENTS
    (tmp_path / "runs").mkdir()
    _, environment, _ = _sections(_page(tmp_path, tmp_path / "runs"), "no model yet")
    base, *later = _rows(environment)
    assert base[:2] == ["Base", BASE_POINTER] and base[3] == design.AGENTS_PRESET and base[5:] == [""] * 6
    assert [(line[0], line[1], line[3]) for line in later] == [
        (name, agent, preset) for preset, name in zip(design.LADDER[1:], design.LADDER_NAMES[1:]) for agent in LADDER_AGENTS]


def test_a_run_of_the_scheduled_preset_is_the_schedule_arm_of_l5_and_the_deterioration_rows_stay_the_environment_and_the_reference(tmp_path):
    """The arm of L5 plays the agent without the MCP; its reference, the same agent as released on all six big layers, is the
    reference arm of En6, the first place on that cell, and the same agent with the MCP there is the top rung of the ladder."""
    runs = _runs(tmp_path / "runs")
    _run(runs, "causal_scheduled", [_on("scheduled", s, agent="causal_m", diagnostics_server=False) for s in range(4)], {**DEVELOPMENT, "config_name": "scheduled"})
    _run(runs, "causal_reference", [_on("deterioration", s, agent="causal_m", diagnostics_server=False) for s in range(4)],
         {**DEVELOPMENT, "config_name": "deterioration"})
    _run(runs, "causal_deterioration", [_on("deterioration", s, agent="causal_m") for s in range(4)], {**DEVELOPMENT, "config_name": "deterioration"})
    _, environment, ablations = _sections(_page(tmp_path, runs))
    l5, = _lines(ablations, "L5")
    assert l5[3:7] == ["the fixed schedule of level drops against the random hazard rates", CD, "m", "scheduled"] and l5[13] == "`causal_scheduled`"
    reference, = (line for line in _lines(ablations, "En6") if line[4] == CD and line[13] == "`causal_reference`")
    assert reference[3].startswith(design.REFERENCE_ARM) and reference[6] == "deterioration"
    deterioration, = (line for line in _lines(environment, "+ deterioration and failure") if line[1] == CD_MCP)
    assert deterioration[3] == "deterioration" and deterioration[10] == "`causal_deterioration`"


def test_a_trained_agent_and_a_row_without_the_mcp_beside_layer_0_are_left_out_with_their_reasons(tmp_path, monkeypatch):
    """The tables of training read a trained agent, any training variant of run.json but the agent as released, and no table
    reads a row without the CausalDiagnosis MCP on a middle rung of the ladder, so the page puts neither on a line and INDEX.json
    names each with its reason, while the rows with the MCP on a later rung are lines of the environment table. A later trained run on layer 0 leaves the main row to the agent as released. The trained run on the deterioration row
    names the fold its model holds out and plays that fold, here every start in fold 1 by the rule of the
    test."""
    monkeypatch.setattr(aggregate.training_tables, "fold_of", lambda combination, asset_id=None, split=None: 1)
    runs = _runs(tmp_path / "runs")
    _run(runs, "causal_mcp_gepa", [_main_row(s, agent="causal_m", **{"pass": False}) for s in range(4)],
         {**DEVELOPMENT, design.TRAINING_VARIANT_KEY: "gepa", "started": "2026-11-02T15:00:00"})
    _run(runs, "causal_mcp_ft", [_on(design.TRAINING_PRESET, s, agent="causal_m") for s in range(4)],
         {**DEVELOPMENT, "config_name": design.TRAINING_PRESET, design.TRAINING_VARIANT_KEY: "ft", design.TRAINING_FOLD_KEY: 1})
    _run(runs, "pe_mcp_deterioration", [_on(design.TRAINING_PRESET, s, agent="plan_execute_m") for s in range(4)],
         {**DEVELOPMENT, "config_name": design.TRAINING_PRESET})
    _run(runs, "pe_bare_priced", [_on("priced", s, agent="plan_execute_m", diagnostics_server=False) for s in range(4)], {**DEVELOPMENT, "config_name": "priced"})
    text = _page(tmp_path, runs)
    main, environment, _ = _sections(text)
    assert _lines(main, CD_MCP)[0][9] == "`causal_mcp`" and _lines(main, CD_MCP)[0][5].startswith("1.000 [")
    for name in ("causal_mcp_gepa", "causal_mcp_ft", "pe_bare_priced"):
        assert f"`{name}`" not in text
    top, = (line for line in _lines(environment, "+ deterioration and failure") if line[1] == PE_MCP)
    assert top[10] == "`pe_mcp_deterioration`"
    left_out = _left_out(tmp_path)
    assert left_out["causal_mcp_gepa"] == left_out["causal_mcp_ft"] == leaderboard.TRAINED
    assert left_out["pe_bare_priced"] == leaderboard.WITHOUT_MCP_BESIDE_LAYER_0
    # INDEX.json shows each reason to a reader without the code, so each names what its run is
    assert leaderboard.TRAINED.startswith("a trained agent")
    assert leaderboard.WITHOUT_MCP_BESIDE_LAYER_0.startswith("a row without the CausalDiagnosis MCP on a later layer")
    assert not [line for line in _rows(environment) if line[1] in (PE, AAT, CD)]


def test_a_running_run_of_a_trained_agent_is_no_untrained_row_still_running(tmp_path):
    """A run in progress has no run.json, so its training variant is read from the ledger that the driver writes before the first
    episode. A running run of a trained agent is left out as its finished run would be, so the row of the agent as released
    says not run, while a running run without a ledger or with the variant as released in it is that row still running, and the
    index records the variant of every run in progress for the rebuild of verify_results_folder.py."""
    runs = tmp_path / "runs"
    _run(runs, "causal_bare", [_main_row(s, agent="causal_m", diagnostics_server=False) for s in range(4)], DEVELOPMENT)
    _running(runs, design.AGENTS_PRESET, "plan_execute_m", variant="gepa")
    _running(runs, design.AGENTS_PRESET, "agent_as_tool_m", variant=design.AS_RELEASED)
    _running(runs, design.AGENTS_PRESET, "causal_m")
    main, _, _ = _sections(_page(tmp_path, runs))
    assert _lines(main, PE_MCP)[0][4] == "not run" and _lines(main, AAT_MCP)[0][4] == "still running" and _lines(main, CD_MCP)[0][4] == "still running"
    running = {run["agent"]: run for run in _index(tmp_path)["running"]}
    assert {agent: run["training_variant"] for agent, run in running.items()} == {"plan_execute_m": "gepa", "agent_as_tool_m": design.AS_RELEASED, "causal_m": design.AS_RELEASED}
    assert leaderboard.classify(running["plan_execute_m"]) is None and leaderboard.classify(running["agent_as_tool_m"]) == leaderboard.Slot("main", 3)


def test_a_run_of_the_mcp_follower_is_a_reference_that_the_page_leaves_out(tmp_path):
    """The MCP follower of design.POLICY_TABLE_POLICIES needs no model and plays AssetOpsBench's servers with the CausalDiagnosis MCP.
    Of the tables that select agents POLICIES alone reads its runs, so the page sets a run of it on no line and INDEX.json gives the
    reason. The control: without its entry in REFERENCES the label is unknown to the page and stops the script."""
    assert [label for label, _ in design.POLICY_TABLE_POLICIES][len(design.REFERENCE_POLICIES):] == ["mcp_follower"] and "mcp_follower" in leaderboard.REFERENCES
    runs = _runs(tmp_path / "runs")
    _run(runs, "follower_policy", [_main_row(s, agent="mcp_follower", model="", diagnostics_server=True, **{"pass": s == 0}) for s in range(4)],
         {**REFERENCE, "servers": list(METERED_SERVERS)})
    text = _page(tmp_path, runs)
    assert "`follower_policy`" not in text and "mcp_follower" not in text and "MCP follower" not in text
    assert _left_out(tmp_path)["follower_policy"] == "a reference policy that the page leaves out"
    assert leaderboard.classify(_as_run(_main_row(0, agent="mcp_follower", model=""))) is None
    # the lines of the follower in grid_agents.txt, one per rung of the ladder, resolve to runs that the page leaves out as well
    lines = [line for line in pending_lines.grid_lines(CLUSTER_JOBS / "grids" / "grid_agents.txt") if pending_lines.expected(line)["agent"] == "mcp_follower"]
    assert [pending_lines.expected(line)["config_name"] for line in lines] == list(design.LADDER)
    for line in lines:
        fields = pending_lines.expected(line)
        run = _as_run({"agent": "mcp_follower", "model": "", "config": fields["config_name"], **{k: fields[k] for k in GRID_SETTINGS}})
        assert leaderboard.classify(run) is None and leaderboard.left_out(run, {}, [], {}) == "a reference policy that the page leaves out"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(leaderboard, "REFERENCES", tuple(label for label in leaderboard.REFERENCES if label != "mcp_follower"))
        with pytest.raises(ValueError, match="unknown agent label 'mcp_follower'"):
            leaderboard.framework_of("mcp_follower", "")


def test_a_run_of_the_random_policy_is_a_reference_that_the_page_leaves_out(tmp_path):
    """The random policy is a reference policy of design.REFERENCE_POLICIES and needs no model. The page shows the oracle and the
    declaring policy under the main table and no other reference, so a run of the random policy, or of the oracle on a row of
    the ladder after layer 0, sits on no line and INDEX.json gives the reason."""
    assert "random" in [label for label, _ in design.REFERENCE_POLICIES] and "random" in leaderboard.REFERENCES
    runs = _runs(tmp_path / "runs")
    _run(runs, "random_policy", [_main_row(s, agent="random", model="", **{"pass": s == 0}) for s in range(4)], REFERENCE)
    _run(runs, "oracle_deterioration", [_on(design.TRAINING_PRESET, s, agent="oracle", model="") for s in range(4)],
         {**REFERENCE, "config_name": design.TRAINING_PRESET})
    text = _page(tmp_path, runs)
    assert "`random_policy`" not in text and "`oracle_deterioration`" not in text and "in the run folder `oracle`" in text
    left_out = _left_out(tmp_path)
    assert left_out["random_policy"] == left_out["oracle_deterioration"] == "a reference policy that the page leaves out"
    assert leaderboard.classify(_as_run(_main_row(0, agent="random", model=""))) is None
    # an anchored run is on no line, and no table of the paper reports it
    assert leaderboard.left_out(_as_run(_main_row(0, agent="plan_execute_m", config="anchored")), None, None, {}) == "a run of the preset anchored, which the paper leaves out"
    assert (leaderboard.left_out(_as_run(_main_row(0, agent="plan_execute_m", config=design.HEALTHY_CONTROL_PRESET)), None, None, {})
            == f"a run of the preset {design.HEALTHY_CONTROL_PRESET}, which the table HEALTHY_CONTROL reports apart")


def test_an_element_varied_where_no_ablation_of_the_paper_varies_it_stops_the_script():
    for name, row, problem in (
            ("guidance_priced", _on("priced", 0, agent="plan_execute_m", diagnostics_server=False, guidance_level=1), "varies En7 on the preset 'priced'"),
            ("seq_rules", _on("repair_rules", 0, agent="causal_seq_m"), "varies Ag9 on the preset 'repair_rules'"),
            ("two_at_once", _main_row(0, agent="causal_nobranch_m", guidance_level=1), "varies Ag1 and En7 at once")):
        with pytest.raises(ValueError, match=problem):
            leaderboard.classify(_as_run(row, name))
    # one element varied on layer 0 is that element's arm, named without a row
    assert leaderboard.classify(_as_run(_main_row(0, agent="plan_execute_m", guidance_level=1), "g1")) == leaderboard.Slot("ablation", 1, "En7", "guidance level 1")


def test_a_pass_rate_prints_the_digits_of_the_share_and_not_those_of_a_rounded_cell(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "pe_mcp", [_main_row(s, agent="plan_execute_m", **{"pass": s < 38}) for s in range(47)], DEVELOPMENT)
    main, _, _ = _sections(_page(tmp_path, runs))
    assert _lines(main, PE_MCP)[0][5].startswith("0.809 [")  # 38 of 47 is 0.80851, which a cell of four digits would print as 0.808


def test_the_pass_rate_carries_the_interval_clustered_by_start_and_not_one_that_takes_episodes_for_independent(tmp_path):
    """The episodes of one start share its recorded block, so the interval of the pass rate comes from a bootstrap that
    resamples whole blocks (stats.cluster_proportion), here with the 50 resamples that _page gives tables_from_runs/ and its seed 0.
    The episodes of one block move together, so a run on a single block has an interval of width zero, where an interval that
    takes the episodes for independent, as Wilson's does, is wide."""
    blocks = [json.dumps(levels) for levels in ([3, 100, 0, 130], [20, 100, 0, 130], [100, 90, 0, 130], [100, 100, 1, 130])]
    passes = [True, True, True, False, False, False, True, False]  # two episodes per block, a share of one half
    runs = tmp_path / "runs"
    _run(runs, "pe_mcp", [_main_row(s, agent="plan_execute_m", block=blocks[s // 2], **{"pass": passes[s]}) for s in range(8)], DEVELOPMENT)
    _run(runs, "aat_mcp", [_main_row(s, agent="agent_as_tool_m", **{"pass": s == 0}) for s in range(4)], DEVELOPMENT)
    main, _, _ = _sections(_page(tmp_path, runs))
    assert "| Pass rate (95% interval clustered by start) |" in main
    share = stats.cluster_proportion(passes, [blocks[s // 2] for s in range(8)], None, 50, 0)
    low, high = (f"{round(share[end], 6):.3f}" for end in ("low", "high"))
    assert share["low"] < share["proportion"] < share["high"]
    assert _lines(main, PE_MCP)[0][5] == f"0.500 [{low}, {high}]"
    assert _lines(main, AAT_MCP)[0][5] == "0.250 [0.250, 0.250]"


def test_the_cutoff_and_the_latest_start_choose_within_one_line_and_never_across_the_two_tables(tmp_path):
    """The main row of an agent with the MCP and its lines of the environment table carry one agent label, and the page takes
    the latest start among the counted runs of a line. A later run on a row of the ladder takes no main row and is no candidate
    of it, a run on a row of the ladder fills no main row whose run on layer 0 started before the cutoff, a run on layer 0
    fills no line of the ladder, and of two counted runs of one line the later start counts and the earlier is its candidate."""
    before, after, later = "2026-11-02T11:00:00", "2026-11-02T13:00:00", "2026-11-02T15:00:00"  # around the cutoff of plan_execute in CUTOFF
    priced_line = f"+ priced reads and writes, {PE_MCP}"

    def page(name, on_base, on_priced, again=None):
        root = tmp_path / name
        _run(root / "runs", "base_run", [_main_row(s, agent="plan_execute_m") for s in range(4)], {**DEVELOPMENT, "started": on_base})
        _run(root / "runs", "priced_run", [_on("priced", s, agent="plan_execute_m", **{"pass": s < 2}) for s in range(4)],
             {**DEVELOPMENT, "config_name": "priced", "started": on_priced})
        if again:
            _run(root / "runs", "base_again", [_main_row(s, agent="plan_execute_m", **{"pass": s < 3}) for s in range(4)],
                 {**DEVELOPMENT, "started": again})
        main, environment, _ = _sections(_page(root, root / "runs"))
        priced, = (line for line in _lines(environment, "+ priced reads and writes") if line[1] == PE_MCP)
        section = _index(root)
        return _lines(main, PE_MCP)[0], priced, {line["line"]: line["candidates"] for line in section["lines"]}, section["left_out"]

    main, priced, candidates, left_out = page("priced_later", after, later)
    assert main[9] == "`base_run`" and main[5].startswith("1.000 [") and priced[10] == "`priced_run`" and priced[6].startswith("0.500 [")
    assert candidates == {PE_MCP: [], priced_line: []} and left_out == []
    main, priced, candidates, left_out = page("base_too_early", before, after)
    assert main[4] == "not run" and priced[10] == "`priced_run`"
    assert left_out == [{"run": "base_run", "reason": f"started {before}, before the cutoff 2026-11-02T12:00:00 of plan_execute"}]
    main, priced, candidates, left_out = page("priced_too_early", after, before)
    assert main[9] == "`base_run`" and priced[5] == "not run"
    assert left_out == [{"run": "priced_run", "reason": f"started {before}, before the cutoff 2026-11-02T12:00:00 of plan_execute"}]
    main, priced, candidates, left_out = page("base_again", after, after, later)
    assert main[9] == "`base_again`" and main[5].startswith("0.750 [") and priced[10] == "`priced_run`"
    assert candidates == {PE_MCP: ["base_run"], priced_line: []}
    assert left_out == [{"run": "base_run", "reason": "an earlier start of the same line, base_again counts"}]


ROW_AGENTS = {prefix for prefix, _, _ in design.AGENT_ROWS}
GRID_SETTINGS = ("clock_mode", "guidance_level", "verifier_scope", "ranking_model", "prices", "diagnostics_server", "abstention_rule")


def _varies_nothing(fields):
    """Whether a grid line plays an agent of the six rows at every level of its preset and of the driver, without a switch of
    the CausalDiagnosis agent: a line of the main table, of the environment table or of a preset of its own arm
    (design.ARM_PRESETS), and not an arm that varies a setting or a switch."""
    preset = load_config(fields["config_name"])[1]
    driver = run_episodes.build_parser().parse_args(["--agent", fields["agent"]])
    return (fields["agent"] in ROW_AGENTS
            and (fields["clock_mode"], fields["guidance_level"], fields["verifier_scope"]) == (preset.clock_mode, preset.guidance_level, preset.verifier_scope)
            and (fields["prices"], fields["ranking_model"], fields["abstention_rule"]) == ("default", driver.ranking_model, driver.abstention_rule)
            and fields.get("causal_options") in (None, run_episodes.causal_options(driver)))


def _place(fields):
    """Where experiment_tables.py puts a grid line that varies nothing: the reason of the healthy control or of a token twin of a clock preset,
    its row of the main table on
    layer 0, the arm of its own preset, its line of the environment table for an agent with the MCP on a later row of the
    ladder, the reference arm of the first supporting ablation placed on its preset and row, and otherwise the reason of a row
    without the MCP beside layer 0."""
    preset = fields["config_name"]
    if preset == design.HEALTHY_CONTROL_PRESET:
        return f"a run of the preset {preset}, which the table HEALTHY_CONTROL reports apart"
    if preset in design.TOKEN_PRESETS:
        return (f"a run of the preset {preset}, the twin of {design.TOKEN_PRESETS[preset]} in the generated_tokens mode, a second "
                "condition of the clock that the page leaves out")
    row = [(prefix, mcp) for prefix, mcp, _ in design.AGENT_ROWS].index((fields["agent"], fields["diagnostics_server"]))
    if preset == design.AGENTS_PRESET:
        return leaderboard.Slot("main", row)
    if preset in design.ARM_PRESETS:
        return leaderboard.Slot("ablation", row, *design.ARM_PRESETS[preset])
    if design.AGENT_ROWS[row] in design.LADDER_ROWS:
        return leaderboard.Slot("environment", row, "", design.LADDER_NAMES[design.LADDER.index(preset)])
    places = [name for name, (place, rows) in design.ABLATION_PLACES.items() if place == preset and design.AGENT_ROWS[row] in rows]
    if places:
        return leaderboard.Slot("ablation", row, places[0], f"{design.REFERENCE_ARM}, on the row {design.LADDER_NAMES[design.LADDER.index(preset)]}")
    return leaderboard.WITHOUT_MCP_BESIDE_LAYER_0


def test_every_line_of_the_grid_parses_names_no_removed_switch_and_a_line_that_varies_nothing_finds_its_place():
    """The run lines of CausalDiagnosis/cluster_jobs/grids/grid_agents.txt as run_episodes/ resolves them. Every line parses, and none names the unread table
    or the rule only switch, which left with their ablations. A line that varies nothing lands where experiment_tables.py puts it, on
    a row of the main table, a line of the environment table, an arm or the reference arm of a supporting ablation, or is left
    out as a row without the MCP beside layer 0 or as a run of the healthy control, which a table of its own reports.
    The lines of the ablation arms wait for the layers that the design gives them, which the grid follows once it
    is confirmed, so their place is not asserted here."""
    lines = pending_lines.grid_lines(CLUSTER_JOBS / "grids" / "grid_agents.txt")
    assert not [line for line in lines if "--diagnostics-table" in line or "--causal-rule-only" in line]
    resolved = {line: pending_lines.expected(line) for line in lines}
    unvaried = {line: fields for line, fields in resolved.items() if _varies_nothing(fields)}
    placed = {}
    for line, fields in unvaried.items():
        label = run_episodes.agent_label_of(fields["agent"], "m", fields.get("causal_options"))
        run = _as_run({"agent": label, "model": "m", "config": fields["config_name"], **{k: fields[k] for k in GRID_SETTINGS}})
        placed[line] = leaderboard.classify(run) or leaderboard.left_out(run, {}, [], {})
    assert placed == {line: _place(fields) for line, fields in unvaried.items()}
    assert {"main", "environment"} <= {place.kind for place in placed.values() if isinstance(place, leaderboard.Slot)}


def test_the_environment_rows_are_the_ladder_of_design_py_with_its_names():
    """The rows of the environment table are the presets of design.LADDER under the names of design.LADDER_NAMES,
    one big layer added per row in the order of the environment figure."""
    assert leaderboard.ENVIRONMENT_ROWS == tuple(zip(design.LADDER, design.LADDER_NAMES))
    assert design.LADDER == ("base", "priced", "real_time", "advanced_real_time", "repair_rules", "deterioration")
    assert design.LADDER_NAMES == ("Base", "+ priced reads and writes", "+ real-time clock", "+ advanced real-time clock", "+ repair rules",
                                   "+ deterioration and failure")


def test_every_reader_takes_the_presets_from_design_py():
    """experiment_tables.py is the one place that names the presets, so a preset renamed or regrouped there changes every
    reader: the rows and the places of the ablations of the leaderboard, the preset of En4 in tables_from_runs/, the row presets of the
    driver's guard in episode_launchers.py, run_configuration.py, run_folder_layout.py, servers_of_a_run.py, CausalDiagnosis/utils/csv_rows.py and CausalDiagnosis/utils/json_files.py and the preset of training that gepa_prompt_optimization/ plays by default. Every preset it names for a table
    or an arm is a preset of the configs folder."""
    from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online  # here, so that an import problem of the GEPA driver fails this test alone

    assert leaderboard.ENVIRONMENT_ROWS == tuple(zip(design.LADDER, design.LADDER_NAMES)) and leaderboard.MAIN_ROWS == design.AGENT_ROWS
    assert aggregate.EN4_PRESET == design.AGENTS_PRESET and aggregate.ABLATION_PLACES is design.ABLATION_PLACES
    assert set(design.LADDER) | set(design.ARM_PRESETS) | {design.HEALTHY_CONTROL_PRESET} | set(design.TOKEN_PRESETS) == set(run_configuration.ROW_PRESETS)
    assert gepa_online.build_parser().get_default("config") == design.TRAINING_PRESET
    named = {design.AGENTS_PRESET, design.TRAINING_PRESET, design.HEALTHY_CONTROL_PRESET, *design.LADDER, *design.ARM_PRESETS,
             *(preset for preset, _ in design.ABLATION_PLACES.values())}
    assert all(load_config(preset)[0] == preset for preset in named)


def test_the_pages_print_the_paper_names_of_the_reference_policies_and_never_the_old_ones(tmp_path):
    """The paper names the policy that the code labels oracle the oracle policy, so the leaderboard and aggregate.md print the
    oracle policy and neither page prints perfect information. The code labels stay. The paper names the policy
    labelled do_nothing after its label, do nothing (design.REFERENCE_POLICIES), so that name may appear on a page."""
    text = _page(tmp_path, _runs(tmp_path / "runs"))
    tables = (tmp_path / "aggregate" / "aggregate.md").read_text(encoding="utf-8")
    assert "| Cost / oracle policy cost |" in text
    assert "The oracle policy, which reads the hidden state, is the best of the five reference policies and not a row of this table." in text
    assert "over the cost of the oracle policy on the same seed" in text and "over the cost of the oracle policy on the same seed" in tables
    for page in (text, tables):
        assert "perfect information" not in page.lower()


def test_the_page_opens_as_a_status_board_that_names_the_page_of_the_tables_of_the_paper(tmp_path):
    """The leaderboard counts runs and prints the verifier's pass of each line in its own order, so the line under its title says that
    it is a status board and names the page that holds the tables of the paper, with no run and with runs alike."""
    from CausalDiagnosis.evaluation.result_tables import paper_tables as main_tables

    (tmp_path / "empty").mkdir()
    for folder, runs in (("a", tmp_path / "empty"), ("b", _runs(tmp_path / "runs"))):
        (tmp_path / folder).mkdir()
        lines = _page(tmp_path / folder, runs).splitlines()
        assert lines[:4] == ["# Leaderboard of the hydraulic rig gym", "", leaderboard.STATUS_LINE, ""]
    assert f"`{main_tables.PAGE}`" in leaderboard.STATUS_LINE


def test_the_judge_rate_waits_for_a_verdict_of_every_episode():
    """Once a judge is named, an episode without a verdict counts as a failure of the judge, so the rate has every episode behind it
    before any verdict exists; the page shows it only when RUNS counts a verdict for every episode."""
    run = {"n": "4", "judge_pass_n": "4", "judge_pass": "0.0", "scope_pass": "0.5", "scope_pass_low": "0.25", "scope_pass_high": "0.75",
           "cost_normalized": "1.0", "delay_median": "", "delay_n": "0", "run": "r"}
    assert leaderboard.cells({**run, "judge_verdicts": "0"})[2] == "judge pending"
    assert leaderboard.cells({**run, "judge_verdicts": "4"})[2] == "0.000"


def test_a_call_that_names_its_aggregate_writes_its_page_beside_the_tables_and_no_page_of_the_fork(tmp_path):
    """--out takes LEADERBOARD.md in the parent of --aggregate, where --index takes INDEX.json and where the results folder of the
    fork holds its leaderboard beside its folder of tables, which the default --aggregate reads. So a call on the tables of another
    folder writes its page there, where verify_results_folder.py --results reads it, and leaves the page of the fork as it is. The
    control names its own --out, which takes the page there."""
    runs, results = tmp_path / "runs", RESULTS
    runs.mkdir()
    stamp = (results / "LEADERBOARD.md").stat().st_mtime_ns
    aggregate.main(["--runs", str(runs), "--out", str(tmp_path / "T" / "aggregate"), "--fraction", "1"])
    leaderboard.main(["--aggregate", str(tmp_path / "T" / "aggregate"), "--runs", str(runs)])
    page = tmp_path / "T" / "LEADERBOARD.md"
    assert page.is_file() and "leaderboard" in json.loads((tmp_path / "T" / "INDEX.json").read_text(encoding="utf-8"))
    assert (results / "LEADERBOARD.md").stat().st_mtime_ns == stamp
    written = page.stat().st_mtime_ns
    leaderboard.main(["--aggregate", str(tmp_path / "T" / "aggregate"), "--runs", str(runs), "--out", str(tmp_path / "other" / "LEADERBOARD.md")])
    assert (tmp_path / "other" / "LEADERBOARD.md").is_file() and page.stat().st_mtime_ns == written
    assert (results / "LEADERBOARD.md").stat().st_mtime_ns == stamp
