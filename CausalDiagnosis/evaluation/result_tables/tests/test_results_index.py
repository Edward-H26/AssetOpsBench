"""The results index: on synthetic run folders made with the helpers of test_tables_from_runs.py, every row of every table lists
exactly the episodes that built it (and a paired row the episodes of its reference level), the run folders read carry
their ledger hash, every folder left unread carries its reason, a run on the memory backend enters no table, and
leaderboard.py adds the run it chose for each line with the rule and every counted run it left out with the reason."""
from CausalDiagnosis.utils.paths import PACKAGE, RESULTS
import hashlib
import json

import pytest

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.evaluation.result_tables import leaderboard
from CausalDiagnosis.evaluation.result_tables import pooling_rules
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS
from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_RANKING_MODEL
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import METERED_SERVERS, RIG_SERVER
from CausalDiagnosis.utils.csv_rows import write_rows
from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import _csv, _row, _run
from CausalDiagnosis.evaluation.result_tables.tests.test_leaderboard import CUTOFF, DEVELOPMENT, _main_row, _page, _runs


def _index(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_row_lists_exactly_its_episodes_and_every_unread_folder_its_reason(tmp_path):
    runs = tmp_path / "runs"
    on = _run(runs, "on", [_main_row(s, agent="plan_execute_m") for s in range(4)], DEVELOPMENT)
    _run(runs, "off", [_main_row(s, agent="plan_execute_m", diagnostics_server=False, **{"pass": s % 2 == 0}) for s in range(4)], DEVELOPMENT)
    (on / "ledger.json").write_text('{"status": "finished"}\n', encoding="utf-8")
    unfinished = runs / "static_plan_execute_m_20261102-130000"
    unfinished.mkdir()
    write_rows(unfinished / "scores.csv", [_row(0)], COLUMNS)
    _run(runs / "archive_invalid_runs", "moved", [_row(0)])
    out = tmp_path / "results" / "aggregate"
    tables = aggregate.aggregate(runs, out, resamples=50, fraction="1")
    index = _index(tmp_path / "results" / "INDEX.json")
    assert index["runs_root"] == "runs" and index["leaderboard"] is None and index["paper"] is None
    assert {i["run"]: (i["episodes"], i["ledger_sha256"], i["ledger_status"]) for i in index["inputs"]} == {
        "off": (4, None, None), "on": (4, hashlib.sha256((on / "ledger.json").read_bytes()).hexdigest(), "finished")}
    assert index["excluded"] == [
        {"folder": "archive_invalid_runs", "reason": "archived as invalid, moved and never deleted"},
        {"folder": "static_plan_execute_m_20261102-130000", "reason": "no run.json, so the run has not finished"}]
    read = {f"{run}/seed{s:06d}" for run in ("on", "off") for s in range(4)}
    assert [entry["episodes"] for entry in index["tables"]["RUNS.csv"]] == [sorted(e for e in read if e.startswith(run)) for run in ("off", "on")]
    h2a, = (entry for entry in index["tables"]["AGENTS_pairs.csv"] if entry["key"]["hypothesis"] == "H2a")
    assert h2a["episodes"] == sorted(e for e in read if e.startswith("on/")) and h2a["reference"] == sorted(e for e in read if e.startswith("off/"))
    agents = {entry["key"]["diagnostics_server"]: entry for entry in index["tables"]["AGENTS.csv"]}
    assert agents["False"]["episodes"] == sorted(e for e in read if e.startswith("off/")) and agents["False"]["reference"] is None
    # every row of every table is traced, to episodes the aggregate read, and the row numbers follow the CSV file
    for table, rows in tables:
        entries = index["tables"][f"{table.id}.csv"]
        assert [e["row"] for e in entries] == list(range(1, len(rows) + 1))
        assert all(e["episodes"] is not None and set(e["episodes"]) <= read and set(e["reference"] or []) <= read for e in entries), table.id
    # the code that computes the tables and the code that renders them, the lists analysis and rendering of experiment_tables.py, apart
    assert set(index["code"]) == {"analysis", "rendering"} and "CausalDiagnosis/evaluation/result_tables/paper_tables/__init__.py" in index["code"]["rendering"]
    assert index["code"]["analysis"]["CausalDiagnosis/evaluation/result_tables/tables_from_runs/__init__.py"] == hashlib.sha256((PACKAGE / "evaluation" / "result_tables" / "tables_from_runs" / "__init__.py").read_bytes()).hexdigest()
    assert index["root_kind"] == "legacy" and index["judges_left_out"] == [] and index["record"] == {}


def test_the_leaderboard_records_its_choice_the_rule_and_every_counted_run_it_left_out(tmp_path):
    _page(tmp_path, _runs(tmp_path / "runs"))
    index = _index(tmp_path / "INDEX.json")
    section = index["leaderboard"]
    assert section["since"] == dict([CUTOFF.split("=", 1)])
    lines = {line["line"]: line for line in section["lines"]}
    assert lines["Plan-and-Execute + CausalDiagnosis MCP"] == {
        "model": "m", "line": "Plan-and-Execute + CausalDiagnosis MCP", "run": "pe_mcp", "candidates": [],
        "rule": {"cutoff": "2026-11-02T12:00:00", "episode_set": "combinations", "choice": "the latest start among the counted runs of the set"}}
    assert lines["the oracle policy"]["run"] == "oracle" and lines["the oracle policy"]["rule"]["cutoff"] is None
    assert {line["run"] for line in section["lines"]} == {"pe_mcp", "aat_bare", "aat_mcp", "causal_bare", "causal_mcp", "oracle", "len1"}
    assert section["left_out"] == [{"run": "aat_timed", "reason": "a run of the preset timed, which no table of the page reads since no row and no supporting ablation plays this preset"},
                                   {"run": "pe_mcp_old", "reason": "started 2026-11-02T11:00:00, before the cutoff 2026-11-02T12:00:00 of plan_execute"}]
    assert {"folder": "base_causal_nobranch_m_20261102-130000", "reason": "no run.json, so the run has not finished"} in index["excluded"]


def test_no_row_of_any_table_mixes_a_trained_run_with_the_agent_without_training(tmp_path):
    """A trained agent and the agent without training carry one agent label, so the training variant of run.json alone keeps their
    episodes apart in every row that a reader follows back to its folders."""
    runs = _runs(tmp_path / "runs")
    _run(runs, "pe_gepa", [_main_row(s, agent="plan_execute_m", **{"pass": s < 2}) for s in range(4)], {**DEVELOPMENT, design.TRAINING_VARIANT_KEY: "gepa"})
    _page(tmp_path, runs)
    tables = _index(tmp_path / "INDEX.json")["tables"]
    for name, entries in tables.items():
        for entry in entries:
            played = {episode.split("/")[0] for episode in entry["episodes"] or []}
            assert "pe_gepa" not in played or played == {"pe_gepa"}, (name, entry["row"])
    own = [f"pe_gepa/seed{s:06d}" for s in range(4)]
    assert [e["episodes"] for e in tables["AGENTS.csv"] if e["key"].get("training_variant") == "gepa"] == [own]
    assert [e["key"]["training_variant"] for e in tables["RUNS.csv"] if e["episodes"] == own] == ["gepa"]


def test_a_run_on_the_memory_backend_enters_no_table_and_the_index_names_it_with_the_reason(tmp_path):
    """Every number of the paper comes from the real servers. The driver writes run.json backend
    kind memory for --servers aob --store memory, couchdb for --store couchdb and no backend for the rig's action server,
    so only the first is refused, by the tables, the pages and the leaderboard alike."""
    runs = tmp_path / "runs"
    _run(runs, "rig", [_row(s, agent="plan_execute_m") for s in range(2)])
    _run(runs, "couchdb", [_row(s, agent="plan_execute_m") for s in range(2)], {"backend": {"kind": "couchdb", "url": "http://localhost:5984"}})
    _run(runs, "memory", [_row(s, agent="plan_execute_m", **{"pass": False}) for s in range(2)], {"backend": {"kind": "memory"}})
    results = tmp_path / "results"
    aggregate.aggregate(runs, results / "aggregate", resamples=50, fraction="1")
    leaderboard.main(["--aggregate", str(results / "aggregate"), "--out", str(results / "LEADERBOARD.md"), "--runs", str(runs)])
    index = _index(results / "INDEX.json")
    assert [entry["run"] for entry in index["inputs"]] == ["couchdb", "rig"]
    assert index["excluded"] == [{"folder": "memory", "reason": pooling_rules.MEMORY_BACKEND}] and pooling_rules.MEMORY_BACKEND.endswith("comes from the real servers over CouchDB")
    assert [row["run"] for row in _csv(results / "aggregate" / "RUNS.csv")] == ["couchdb", "rig"]
    traced = [name for entries in index["tables"].values() for entry in entries for name in (entry["episodes"] or []) + (entry["reference"] or [])]
    assert traced and not [name for name in traced if name.startswith("memory/")]
    section = index["leaderboard"]
    assert "memory" not in {line["run"] for line in section["lines"]} | {line["run"] for line in section["left_out"]}
    assert "Run folders under `runs`: 2;" in (results / "aggregate" / "aggregate.md").read_text(encoding="utf-8")
    for page in ("README.md", "LEADERBOARD.md", "aggregate/aggregate.md"):
        assert "`memory`" not in (results / page).read_text(encoding="utf-8"), page


def test_runs_of_the_retired_gepa_mode_and_model_agents_on_a_row_over_the_rig_leave_every_table(tmp_path):
    """A run of the retired mode gepa and a model agent on a row
    preset over the rig's action server join the memory backend runs among the folders that no table reads, while a
    reference policy over the rig, a model agent over AssetOpsBench's servers and a model agent on a preset outside the
    rows stay, each with the surface its run.json names."""
    runs = tmp_path / "runs"
    _run(runs, "gepa", [_row(0, agent="gepa_m")], {"mode": "gepa"})
    _run(runs, "rig_row", [_row(0, config="real_time", agent="plan_execute_m")], {"config_name": "real_time", "servers": [RIG_SERVER]})
    _run(runs, "rig_oracle", [_row(0, config="real_time", agent="oracle", model="")],
         {"config_name": "real_time", "agent": "oracle", "model": None, "servers": [RIG_SERVER]})
    _run(runs, "aob_row", [_row(0, config="real_time", agent="plan_execute_m")], {"config_name": "real_time", "servers": list(METERED_SERVERS)})
    _run(runs, "rig_static", [_row(0, agent="plan_execute_m")], {"servers": [RIG_SERVER]})
    # the scheduled arm is a row of the paper too, so a model agent over the rig's surface on it stays out as well
    _run(runs, "rig_scheduled", [_row(0, config="scheduled", agent="plan_execute_m")], {"config_name": "scheduled", "servers": [RIG_SERVER]})
    out = tmp_path / "results" / "aggregate"
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, out, resamples=50, fraction="1")}
    index = _index(tmp_path / "results" / "INDEX.json")
    assert sorted(i["run"] for i in index["inputs"]) == ["aob_row", "rig_oracle", "rig_static"]
    assert index["excluded"] == [{"folder": "gepa", "reason": pooling_rules.RETIRED_GEPA_MODE},
                                 {"folder": "rig_row", "reason": pooling_rules.RIG_SURFACE_ROW},
                                 {"folder": "rig_scheduled", "reason": pooling_rules.RIG_SURFACE_ROW}]
    assert {r["run"]: r["surface"] for r in tables["RUNS"]} == {"aob_row": "aob", "rig_oracle": "rig", "rig_static": "rig"}


def test_the_rows_that_refuse_the_rig_surface_are_the_rows_of_the_environment_table_and_the_arms_that_play_on_them():
    assert ({preset for preset, _ in leaderboard.ENVIRONMENT_ROWS} | set(design.ARM_PRESETS) | {design.HEALTHY_CONTROL_PRESET} | set(design.TOKEN_PRESETS)
            == set(pooling_rules.ROW_PRESETS))


# run.json of a finished run as run_episodes/ writes it, on the real-time clock row: the fields that name its condition, its
# seeds and the commit it played
CONDITION = {"mode": "gym", "agent_label": "plan_execute_m", "model": "m", "config_name": "real_time", "clock_mode": "generation_time",
             "servers": list(METERED_SERVERS), "diagnostics_server": True, "guidance_level": 2, "verifier_scope": "writes", "prices": "default",
             "ranking_model": DEFAULT_RANKING_MODEL, "abstention_rule": "decision", "abstention_cut": None}
COMMIT = "3" * 40


def _finished(seeds, **fields):
    return {**CONDITION, "seeds": list(seeds), "starts": "sampled", "master_seed": 42, "code": {"commit": COMMIT}, **fields}


SHARED = ("an earlier finished run of the condition of again (started 2026-12-02T00:00:00), which plays 2 of its seeds again (2, 3), so the "
          "later start alone counts, as the leaderboard chooses, and no table pools two runs of one line")


def test_of_two_finished_runs_of_one_condition_on_a_shared_seed_the_later_start_counts_and_the_earlier_is_named(tmp_path, capsys):
    """A line played again, as when a grid is continued without pending_grid_lines.py, would double n and the
    pairs of every table that pools the two runs, so the later start counts, as the leaderboard chooses, the earlier run enters no
    table, INDEX.json names it with both folders and the seeds they share, and tables_from_runs/ prints the same. A run of the
    combinations set beside one of the evaluation set on other seeds, a run that differs in one setting of the condition, the two
    online loops of GEPA, which differ in their gepa block alone, and a run.json without a seeds list, which is compared on its
    commit alone, read together."""
    runs = tmp_path / "runs"
    _run(runs, "first", [_main_row(s, agent="plan_execute_m") for s in range(4)], _finished(range(4)))
    _run(runs, "again", [_main_row(s, agent="plan_execute_m", **{"pass": False}) for s in range(2, 6)], _finished(range(2, 6), started="2026-12-02T00:00:00"))
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    index = _index(tmp_path / "INDEX.json")
    assert [i["run"] for i in index["inputs"]] == ["again"] and [r["run"] for r in tables["RUNS"]] == ["again"]
    assert index["excluded"] == [{"folder": "first", "reason": SHARED}]
    agents, = tables["AGENTS"]
    assert (agents["n"], agents["pass"]) == (4, 0.0)
    assert pooling_rules.pooling_left_out(runs) == {"first": SHARED} and [p.name for p in pooling_rules.folders_read(runs)] == ["again"]
    aggregate.main(["--runs", str(runs), "--out", str(tmp_path / "again" / "aggregate"), "--readme", str(tmp_path / "again" / "README.md"),
                    "--published", str(tmp_path / "again" / "runs"), "--index", str(tmp_path / "again" / "INDEX.json"), "--resamples", "50"])
    assert f"left out first: {SHARED}" in capsys.readouterr().err
    leaderboard.main(["--aggregate", str(tmp_path / "again" / "aggregate"), "--out", str(tmp_path / "again" / "LEADERBOARD.md"), "--runs", str(runs)])
    assert f"- Left out of every table: `first`, {SHARED}" in (tmp_path / "again" / "LEADERBOARD.md").read_text(encoding="utf-8")
    other = tmp_path / "other"
    _run(other, "development", [_main_row(s, agent="plan_execute_m") for s in range(4)], _finished(range(4), starts="combinations", master_seed=None))
    _run(other, "evaluation", [_main_row(s, agent="plan_execute_m") for s in range(4, 8)], _finished(range(4, 8)))
    _run(other, "gepa", [_main_row(s, agent="plan_execute_m") for s in range(4)], _finished(range(4), training_variant="gepa"))
    _run(other, "bare", [_main_row(s, agent="plan_execute_m", diagnostics_server=False) for s in range(4)], _finished(range(4), diagnostics_server=False))
    _run(other, "no_seeds", [_main_row(s, agent="plan_execute_m") for s in range(4)], {**_finished(range(4)), "seeds": None})
    for card in ("verifier", "score_only"):
        _run(other, f"loop_{card}", [_main_row(s, agent="causal_m", gepa_call=1) for s in range(2)],
             _finished(range(2), mode="gepa_online", agent_label="causal_m", gepa={"card": card}))
    assert [p.name for p in pooling_rules.folders_read(other)] == ["bare", "development", "evaluation", "gepa", "loop_score_only", "loop_verifier", "no_seeds"]
    assert pooling_rules.pooling_left_out(other) == {}


def test_the_replicate_runs_of_a_line_share_no_seed_so_all_count_while_a_replicate_played_again_leaves_its_earlier_run_out(tmp_path):
    """The five runs of a line, one per replicate, play distinct seeds of one condition, so the
    refusal of a shared seed never fires on them and all five read, while a replicate played again shares its seeds with its first
    run, whose earlier start is then left out and named."""
    runs = tmp_path / "runs"
    for r in range(1, 6):
        seeds = range(4 * r, 4 * r + 4)
        _run(runs, f"r{r}", [_main_row(s, agent="plan_execute_m") for s in seeds], _finished(seeds, replicate=r, started=f"2026-12-01T0{r}:00:00"))
    assert pooling_rules.pooling_left_out(runs) == {} and [p.name for p in pooling_rules.folders_read(runs)] == [f"r{r}" for r in range(1, 6)]
    _run(runs, "r3_again", [_main_row(s, agent="plan_execute_m") for s in range(12, 16)], _finished(range(12, 16), replicate=3, started="2026-12-02T00:00:00"))
    left_out = pooling_rules.pooling_left_out(runs)
    assert list(left_out) == ["r3"] and left_out["r3"].startswith("an earlier finished run of the condition of r3_again")


def test_the_runs_of_another_commit_than_the_latest_start_enter_no_table_and_are_named_with_both_commits(tmp_path, capsys):
    """Runs of different commits are never pooled, so beside a line played again on
    a fixed commit the runs of the old commit enter no table, INDEX.json and the message name each with both folders and both
    commits, and a run that recorded no commit reads as none recorded. Two runs of one commit on other seeds read together."""
    runs = tmp_path / "runs"
    fixed = "1" * 40
    _run(runs, "pe_bare", [_main_row(s, agent="plan_execute_m", diagnostics_server=False) for s in range(4)], _finished(range(4), diagnostics_server=False))
    _run(runs, "pe_server", [_main_row(s, agent="plan_execute_m") for s in range(4)], _finished(range(4)))
    _run(runs, "pe_server_fixed", [_main_row(s, agent="plan_execute_m") for s in range(4)], _finished(range(4), code={"commit": fixed}, started="2026-12-03T00:00:00"))
    old = (f"played commit {COMMIT} while pe_server_fixed (the latest start, 2026-12-03T00:00:00) played {fixed}, and runs of different "
           "commits are never pooled")
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    index = _index(tmp_path / "INDEX.json")
    assert [i["run"] for i in index["inputs"]] == ["pe_server_fixed"] and [r["run"] for r in tables["RUNS"]] == ["pe_server_fixed"]
    assert index["excluded"] == [{"folder": "pe_bare", "reason": old}, {"folder": "pe_server", "reason": old}]
    aggregate.main(["--runs", str(runs), "--out", str(tmp_path / "again" / "aggregate"), "--readme", str(tmp_path / "again" / "README.md"),
                    "--published", str(tmp_path / "again" / "runs"), "--index", str(tmp_path / "again" / "INDEX.json"), "--resamples", "50"])
    err = capsys.readouterr().err
    assert f"left out pe_bare: {old}" in err and f"left out pe_server: {old}" in err
    leaderboard.main(["--aggregate", str(tmp_path / "again" / "aggregate"), "--out", str(tmp_path / "again" / "LEADERBOARD.md"), "--runs", str(runs)])
    page = (tmp_path / "again" / "LEADERBOARD.md").read_text(encoding="utf-8")
    assert f"- Left out of every table: `pe_bare`, {old}" in page and f"- Left out of every table: `pe_server`, {old}" in page
    upload = tmp_path / "upload"
    _run(upload, "no_commit", [_main_row(s, agent="agent_as_tool_m") for s in range(4)], _finished(range(4), agent_label="agent_as_tool_m", code={"commit": None}))
    _run(upload, "pe_server", [_main_row(s, agent="plan_execute_m") for s in range(4)], _finished(range(4), started="2026-12-02T00:00:00"))
    assert pooling_rules.pooling_left_out(upload) == {"no_commit": f"played commit none recorded while pe_server (the latest start, 2026-12-02T00:00:00) played {COMMIT}, "
                                                      "and runs of different commits are never pooled"}
    one = tmp_path / "one"
    _run(one, "pe_server", [_main_row(s, agent="plan_execute_m") for s in range(4)], _finished(range(4)))
    _run(one, "pe_server_more", [_main_row(s, agent="plan_execute_m") for s in range(4, 8)], _finished(range(4, 8)))
    assert [p.name for p in pooling_rules.folders_read(one)] == ["pe_server", "pe_server_more"] and pooling_rules.pooling_left_out(one) == {}


def test_a_run_without_a_seeds_list_is_compared_on_its_commit_alone(tmp_path):
    """A run.json older than the seeds key has no seeds to compare, so it reads beside a run of its condition whatever seeds that run
    plays, and the commit rule still applies to it."""
    runs = tmp_path / "runs"
    _run(runs, "pe_server", [_main_row(s, agent="plan_execute_m") for s in range(4)], _finished(range(4)))
    _run(runs, "pe_server_old_json", [_main_row(s, agent="plan_execute_m") for s in range(4)], {**_finished(range(4), started="2026-12-02T00:00:00"), "seeds": None})
    assert pooling_rules.pooling_left_out(runs) == {} and [p.name for p in pooling_rules.folders_read(runs)] == ["pe_server", "pe_server_old_json"]
    _run(runs, "pe_server_other_commit", [_main_row(s, agent="plan_execute_m") for s in range(4)], {**_finished(range(4)), "seeds": None, "code": {"commit": "2" * 40}})
    assert pooling_rules.pooling_left_out(runs) == {"pe_server_other_commit": f"played commit {'2' * 40} while pe_server_old_json (the latest start, 2026-12-02T00:00:00) "
                                                    f"played {COMMIT}, and runs of different commits are never pooled"}


# ----- pooling per asset and by identity, the kinds of a root, the head of a registered table -----

def _meta(identity, started, seeds=range(4), **fields):
    return {**_finished(seeds, started=started), "code": {"commit": None, "identity": identity}, **fields}


def test_runs_of_two_assets_never_meet_in_one_rule(tmp_path):
    """Pooling runs per asset: the latest start of one asset never leaves out a run of another asset."""
    runs = tmp_path / "runs"
    _run(runs, "rig_old", [_main_row(s, agent="plan_execute_m") for s in range(4)], _meta("a" * 64, "2026-12-01T00:00:00", stamp="development"))
    _run(runs, "tiny", [_main_row(s, agent="plan_execute_m") for s in range(4)], _meta("b" * 64, "2026-12-02T00:00:00", stamp="development", asset_id="tiny"))
    assert pooling_rules.pooling_left_out(runs) == {} and pooling_rules.root_kind(runs) == "development"


def test_a_root_with_stamped_and_registered_runs_stops_every_table_tool(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "dev", [_main_row(s, agent="plan_execute_m") for s in range(4)], _meta("a" * 64, "2026-12-01T00:00:00", stamp="development"))
    _run(runs, "reg", [_main_row(s, agent="plan_execute_m") for s in range(4)], _meta("a" * 64, "2026-12-02T00:00:00"))
    with pytest.raises(SystemExit, match="holds runs of record and development runs, reg and dev"):
        pooling_rules.root_kind(runs)
    with pytest.raises(SystemExit, match="holds runs of record"):
        aggregate.aggregate(runs, tmp_path / "out", resamples=20, fraction="1")


def _judge_of_record(monkeypatch, served_name, weights_commit="c" * 40):
    """The judge block of the serving file of record as ledger.serving_of_record returns it, for the test."""
    monkeypatch.setattr(pooling_rules.ledger.serving_endpoints, "serving_of_record", lambda path=None: {"judge": {"served_name": served_name, "weights_commit": weights_commit}})


def _of_record(commit, harness, started):
    """A run.json of a run of record of plan_execute that played commit with the harness list {h.py: harness}."""
    meta = _meta("a" * 64, started, agent="plan_execute", model=None)
    meta["code"] = {"commit": commit, "identity": "a" * 64, "files": {"harness": {"h.py": harness}}}
    return meta


def test_runs_of_record_of_two_groups_stop_the_tables_until_a_commit_is_named_whose_group_the_head_names(tmp_path, monkeypatch, registered_root_numbers):
    """Two runs of record whose shared lists differ fall into two groups, which no table pools: the tool stops and names the groups
    with their commits, a named commit keeps its group and leaves the other run out with the part that differs, and every page opens
    with the commit and the group that were read."""
    monkeypatch.setattr(pooling_rules, "_SELECTED_COMMIT", None)
    _judge_of_record(monkeypatch, "j2")
    runs = tmp_path / "runs"
    right = _of_record("1" * 40, "1", "2026-12-01T00:00:00")
    _run(runs, "right", [_main_row(s, agent="plan_execute_m") for s in range(4)], right)
    _run(runs, "other", [_main_row(s, agent="plan_execute_m", diagnostics_server=False) for s in range(4)], _of_record("2" * 40, "2", "2026-12-02T00:00:00"))
    with pytest.raises(SystemExit, match=f"fall into more than one group.*commits {'1' * 12}.*commits {'2' * 12}"):
        pooling_rules.pooling_left_out(runs)
    pooling_rules.select_commit("1111111")
    assert pooling_rules.pooling_left_out(runs) == {"other": "shared lists and data: not those of the runs of the commit 1111111 that the tables read"}
    aggregate.aggregate(runs, tmp_path / "results" / "aggregate", resamples=20, departure=True, commit="1111111", fraction="1")
    head = f"Runs of record of the asset hyd_1, commits {'1' * 12}, group {pooling_rules.group_key(right)['lists'][:12]}."
    for page in ("aggregate/aggregate.md", "README.md", "MAIN_TABLES.md"):
        assert (tmp_path / "results" / page).read_text(encoding="utf-8").startswith(head + "\n"), page
    assert _index(tmp_path / "results" / "INDEX.json")["parameters"]["commit"] == "1111111"


def _identity_tree(root):
    """A tree of one harness file, one episode file and one cluster script, for the hash of the shared lists."""
    for path, text in (("CausalDiagnosis/experiments/episode_runners/run_episodes/__init__.py", "harness\n"), ("CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py", "episode\n"),
                       ("CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch", "#!/bin/bash\n")):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return root


def test_an_edit_of_a_cluster_script_leaves_the_lists_hash_and_the_identity_equal_and_an_edit_of_the_harness_does_not(tmp_path):
    """The cluster scripts are the bound list cluster, hashed into run.json and pinned by the commit, so an edit of one splits no
    group of the tables and moves no code identity, while an edit of a harness file moves both."""
    root = _identity_tree(tmp_path)

    def key_and_identity():
        files = pooling_rules.ledger.code_identity.tree_files(root)
        return pooling_rules.group_key({"agent": "causal", "code": {"files": files}}), pooling_rules.ledger.identity_of(files, "causal"), files

    before, identity, files = key_and_identity()
    script = files["cluster"]["CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch"]
    (root / "CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch").write_text("#!/bin/bash\n# the job asks for another queue\n", encoding="utf-8")
    after, moved, files = key_and_identity()
    assert (after, moved) == (before, identity) and files["cluster"]["CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch"] != script
    (root / "CausalDiagnosis/experiments/episode_runners/run_episodes/__init__.py").write_text("harness, changed\n", encoding="utf-8")
    changed, changed_identity, _files = key_and_identity()
    assert changed["lists"] != before["lists"] and changed_identity != identity


@pytest.mark.parametrize("clock_mode, pooled", [(pooling_rules.WALL_CLOCK, False), ("declared", True)])
def test_two_runs_granted_other_processors_pool_off_the_wall_clock_and_not_on_it(clock_mode, pooled, monkeypatch):
    """Two runs of record that differ only in the processors granted to their jobs fall into two groups on a wall-clock preset and
    into one on any other clock, and a run outside a Slurm job never shares the group of a cluster run on the wall clock."""
    monkeypatch.setattr(pooling_rules, "_SELECTED_COMMIT", None)

    def meta(granted):
        return {"agent": "causal", "clock_mode": clock_mode, "runs_in_flight": 1, "code": {"commit": "1" * 40, "files": {}}, "granted": granted}

    cluster = {"scheduler": "slurm", "cpus": 4, "memory_per_node_mb": 16384, "memory_per_cpu_mb": None}
    for other in ({**cluster, "cpus": 8}, {"scheduler": None, "cpus": None, "memory_per_node_mb": None, "memory_per_cpu_mb": None}):
        metas = {"a": meta(cluster), "b": meta(other)}
        if pooled:
            assert pooling_rules._grouped(["a", "b"], metas, {}) is None
        else:
            with pytest.raises(SystemExit, match="processors and memory granted to the job"):
                pooling_rules._grouped(["a", "b"], metas, {})


def test_a_wall_clock_run_taken_up_under_another_grant_is_left_out_and_one_off_the_wall_clock_is_not(tmp_path):
    """A job that takes up a run records its grant in the resume event: on a wall-clock preset a grant other than that of run.json
    leaves the run out as two identities, the same grant keeps it, and off the wall clock the grant is never compared."""
    granted = {"scheduler": "slurm", "cpus": 16, "memory_per_node_mb": 98304, "memory_per_cpu_mb": None}

    def refusal(clock_mode, later):
        run_dir = tmp_path / f"{clock_mode}_{later['cpus']}"
        run_dir.mkdir()
        event = {"tool": pooling_rules.ledger.RESUME_TOOL, "at": "2026-12-01T01:00:00", "granted": later}
        (run_dir / pooling_rules.ledger.LEDGER).write_text(json.dumps({"events": [event]}), encoding="utf-8")
        return pooling_rules.registered_refusal({"clock_mode": clock_mode, "granted": granted, "code": {}}, run_dir)

    assert "granted other processors or memory" in refusal(pooling_rules.WALL_CLOCK, {**granted, "cpus": 8})
    assert refusal(pooling_rules.WALL_CLOCK, dict(granted)) is None
    assert refusal("declared", {**granted, "cpus": 8}) is None


def test_a_second_finished_run_of_a_registered_condition_stops_the_tables_and_an_archive_event_lifts_it(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "first", [_main_row(s, agent="plan_execute_m") for s in range(4)], _meta("a" * 64, "2026-12-01T00:00:00", agent="plan_execute", model=None))
    again = _run(runs, "again", [_main_row(s, agent="plan_execute_m") for s in range(4)], _meta("a" * 64, "2026-12-02T00:00:00", agent="plan_execute", model=None))
    with pytest.raises(SystemExit, match="two finished runs of one registered condition share seeds, first and again"):
        pooling_rules.pooling_left_out(runs)
    (again / "ledger.json").write_text(json.dumps({"status": "finished", "events": []}), encoding="utf-8")
    pooling_rules.ledger.append_event(again, pooling_rules.ARCHIVE_TOOL, "played again by mistake")
    assert list(pooling_rules.pooling_left_out(runs)) == ["again"]


def test_a_development_root_gets_the_banner_and_cannot_write_into_the_results_folder(tmp_path):
    """Every page of a development root opens with the banner, no table of it enters the results folder of the fork, and neither
    publish_runs.py nor run_review_pack.py takes it."""
    runs = tmp_path / "runs"
    _run(runs, "dev", [_main_row(s, agent="plan_execute_m") for s in range(4)], _meta("a" * 64, "2026-12-01T00:00:00", stamp="development"))
    with pytest.raises(SystemExit, match="the tables of development runs never enter the results folder"):
        aggregate.aggregate(runs, RESULTS / "tables", resamples=20, fraction="1")
    aggregate.aggregate(runs, tmp_path / "out" / "aggregate", resamples=20, fraction="1")
    for page in ("aggregate/aggregate.md", "README.md", "MAIN_TABLES.md"):
        assert (tmp_path / "out" / page).read_text(encoding="utf-8").startswith(pooling_rules.DEVELOPMENT_BANNER + "\n"), page
    assert (tmp_path / "out" / "main_tables.tex").read_text(encoding="utf-8").startswith(f"% {pooling_rules.DEVELOPMENT_BANNER}\n")
    from CausalDiagnosis.episode_viewer import run_review_pack as review_pack
    from CausalDiagnosis.evaluation.result_tables import publish_runs

    with pytest.raises(SystemExit, match="development runs, which are not results"):
        publish_runs.publish(runs, tmp_path / "published")
    with pytest.raises(SystemExit, match="development runs, which are not results"):
        review_pack.build(runs / "dev", tmp_path / "out", tmp_path / "packs")


def test_runs_without_a_commit_are_compared_on_their_identity(tmp_path):
    """Every commit is None on the cluster, so a development root compares identities, per agent."""
    runs = tmp_path / "runs"
    _run(runs, "old", [_main_row(s, agent="plan_execute_m") for s in range(4)], _meta("a" * 64, "2026-12-01T00:00:00", stamp="development", agent="plan_execute"))
    _run(runs, "new", [_main_row(s, agent="plan_execute_m", diagnostics_server=False) for s in range(4)],
         _meta("b" * 64, "2026-12-02T00:00:00", stamp="development", agent="plan_execute"))
    _run(runs, "causal", [_main_row(s, agent="causal_m") for s in range(4)],
         _meta("c" * 64, "2026-11-30T00:00:00", stamp="development", agent="causal", agent_label="causal_m"))
    left = pooling_rules.pooling_left_out(runs)
    assert list(left) == ["old"] and "that of new, the latest start of its agent" in left["old"]


def test_two_runs_of_one_price_label_and_other_prices_never_pool(tmp_path):
    """The label of a run that names no price table is default whatever prices its preset holds, so prices_sha256 enters the
    condition: two runs of one label and other prices are two conditions and never one, and neither leaves the other out."""
    runs = tmp_path / "runs"
    _run(runs, "one", [_main_row(s, agent="plan_execute_m") for s in range(4)], _meta("a" * 64, "2026-12-01T00:00:00", stamp="development", prices_sha256="1" * 64))
    _run(runs, "two", [_main_row(s, agent="plan_execute_m") for s in range(4)], _meta("a" * 64, "2026-12-02T00:00:00", stamp="development", prices_sha256="2" * 64))
    assert "prices_sha256" in pooling_rules.CONDITION_KEYS and pooling_rules.pooling_left_out(runs) == {}


def test_a_judge_record_of_another_judge_than_the_judge_of_record_is_left_out_and_two_judges_of_record_stop_the_tables(monkeypatch):
    """In a root of runs of record a judge record of another served name or weights commit than the judge of the serving file of record
    is left out and named, and records of the judge of record that two judges wrote (protocol, serving or judging code) stop the tool."""
    episodes = [{"run": "r", "seed": 0, "asset_id": "hyd_1", "judge_record_model": "j1", "judge_record_weights_commit": "c" * 40,
                 "judge_record_identity": "x", "judge_record_protocol": "paper", "judge_pass": True, "task_completion": 1.0,
                 "task_completion_majority": True}]
    assert aggregate.foreign_judges(episodes, "development") == [] and episodes[0]["judge_pass"] is True
    _judge_of_record(monkeypatch, "j2")
    named = aggregate.foreign_judges(episodes, "record")
    assert named == [f"run r, seed 0: judged by j1 at the weights commit {'c' * 12}, not by the judge of record j2 at {'c' * 12}"]
    assert episodes[0]["judge_pass"] == "" and "judge_record_model" not in episodes[0] and "task_completion" not in episodes[0]
    two = [{"run": run, "seed": 0, "judge_record_model": "j2", "judge_record_weights_commit": "c" * 40, "judge_record_identity": identity}
           for run, identity in (("r1", "x"), ("r2", "y"))]
    with pytest.raises(SystemExit, match="more than one judge"):
        aggregate.foreign_judges(two, "record")


def test_a_record_that_the_judge_of_record_wrote_through_its_router_backend_is_kept(tmp_path, monkeypatch):
    """A record written by judge_episode through a backend of router_backend names the served name, which the serving file of record
    holds, so a root of runs of record keeps it, under the exact comparison of the served name."""
    from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.tests.test_episode_judge import RUBRIC_PASS, FakeJudge, episode_folder

    ep = episode_folder(tmp_path / "judged")
    from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import episode_judge as judge
    judge.judge_episode(ep, FakeJudge([RUBRIC_PASS], model_id="tokenrouter/j2"), "single", agent_model="a")
    episodes = [{"run": "r", "seed": 7, "asset_id": "hyd_1", **aggregate.judge_fields({"episode_dir": ep}), "judge_pass": True}]
    _judge_of_record(monkeypatch, "j2", weights_commit=episodes[0]["judge_record_weights_commit"])
    assert aggregate.foreign_judges(episodes, "record") == [] and episodes[0]["judge_record_model"] == "j2"


def test_a_registered_root_plans_the_judge_of_record_for_every_played_episode_of_a_model_agent(monkeypatch):
    """A grid of record names no judge in run.json, so in a root of runs of record the judge of the serving file of record is planned
    for every played gym episode and every static answer of a model agent: an episode without its record is unjudged and its judge
    outcomes count as false, while a reference policy and every other root keep the judge their runs name."""
    _judge_of_record(monkeypatch, "j2")
    def episodes():
        return [{"mode": "gym", "model": "m", "asset_id": "hyd_1", "run_judge_model": "", "judge_pass": ""},
                {"mode": "gym", "model": "", "asset_id": "hyd_1", "run_judge_model": "", "judge_pass": ""},
                {"mode": "static", "model": "m", "asset_id": "hyd_1", "run_judge_model": "", "judge_pass": ""}]
    kept = episodes()
    aggregate.plan_judge_of_record(kept, "development")
    assert [e["run_judge_model"] for e in kept] == ["", "", ""]
    planned = episodes()
    aggregate.plan_judge_of_record(planned, "record")
    assert [e["run_judge_model"] for e in planned] == ["j2", "", "j2"]
    assert aggregate.value(planned[0], "judge_pass") == 0.0 and aggregate.value(planned[1], "judge_pass") is None
    assert aggregate.value(planned[2], "judge_pass") == 0.0


def test_a_copy_scored_again_under_its_own_scope_is_read_in_place_of_its_run_and_its_scoring_list_joins_the_group(tmp_path, monkeypatch):
    """The copy that rescore_under_verifier_scope.py --own-scope wrote is read and the run it came from is left out in its place; a copy whose other
    lists differ from its run's is left out with the reason, and a run of record of the old scoring list beside the copy stops the
    tool, since the copy enters the group with the scoring list it was scored under."""
    monkeypatch.setattr(pooling_rules, "_SELECTED_COMMIT", None)
    runs = tmp_path / "runs"
    lists = {"harness": {"h.py": "1"}, "scoring": {"s.py": "1"}}
    source = _meta("a" * 64, "2026-12-01T00:00:00", agent="plan_execute", model=None)
    source["code"]["files"] = lists
    _run(runs, "run", [_main_row(s, agent="plan_execute_m") for s in range(4)], source)
    copy_meta = {**_meta("b" * 64, "2026-12-01T00:00:00", agent="plan_execute", model=None), "rescored": {"from": "run", "own_scope": True}}
    copy_meta["code"]["files"] = {**lists, "scoring": {"s.py": "2"}}
    copy = _run(runs, "run_own_scope", [_main_row(s, agent="plan_execute_m") for s in range(4)], copy_meta)
    (copy / "ledger.json").write_text(json.dumps({"status": "finished", "events": [{"tool": "rescore_scope.py", "what": "copy"}]}), encoding="utf-8")
    assert pooling_rules.pooling_left_out(runs) == {"run": "scored again under its own scope in run_own_scope, which is read in its place"}
    other_lists = {**copy_meta, "code": {**copy_meta["code"], "files": {"harness": {"h.py": "2"}, "scoring": {"s.py": "2"}}}}
    (copy / "run.json").write_text(json.dumps(other_lists), encoding="utf-8")
    assert pooling_rules.pooling_left_out(runs) == {"run_own_scope": "its lists harness differ from those of the run it came from"}
    (copy / "run.json").write_text(json.dumps(copy_meta), encoding="utf-8")
    beside = _meta("a" * 64, "2026-12-02T00:00:00", agent="plan_execute", model=None, seeds=range(4, 8))
    beside["code"]["files"] = lists
    _run(runs, "beside", [_main_row(s, agent="plan_execute_m") for s in range(4, 8)], beside)
    with pytest.raises(SystemExit, match="fall into more than one group"):
        pooling_rules.pooling_left_out(runs)


def test_a_registered_root_is_refused_under_other_resamples_or_an_open_number_unless_the_departure_is_named(tmp_path, monkeypatch, registered_root_numbers):
    """The tables of registered runs use the registered resamples and seed: other values stop the tool unless the call names the
    departure, which INDEX.json then states in words, and a number of the registered file that the tables read and that is open stops
    the tool by its name."""
    runs = tmp_path / "runs"
    _run(runs, "right", [_main_row(s, agent="plan_execute_m") for s in range(4)], _meta("a" * 64, "2026-12-01T00:00:00", agent="plan_execute", model=None))
    _judge_of_record(monkeypatch, "j2")
    with pytest.raises(SystemExit, match="20 resamples where it registers 10000"):
        aggregate.aggregate(runs, tmp_path / "results" / "aggregate", resamples=20, fraction="1")
    aggregate.aggregate(runs, tmp_path / "results" / "aggregate", resamples=20, departure=True, fraction="1")
    words = _index(tmp_path / "results" / "INDEX.json")["parameters"]["departure"]
    assert words.startswith("these tables depart from the registered analysis: 20 resamples where it registers 10000")
    data = json.loads(aggregate.registered.PATH.read_text(encoding="utf-8"))
    data["intervals"]["min_clusters"] = None
    (tmp_path / "open.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="intervals.min_clusters is not set"):
        aggregate.aggregate(runs, tmp_path / "again" / "aggregate", resamples=20, departure=True, registered_path=tmp_path / "open.json", fraction="1")
