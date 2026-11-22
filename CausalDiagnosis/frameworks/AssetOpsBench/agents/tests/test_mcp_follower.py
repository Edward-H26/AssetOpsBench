"""The MCP follower check of the driver: --agent mcp_follower plays development starts of the base row over AssetOpsBench's servers
behind the metering layer with the CausalDiagnosis MCP and no model. Every episode reads the whole record, submits the first
candidate root cause of the ranking citing values that the released rows hold, orders the repair of exactly the components that the
candidate puts off their healthy level, and finishes, so the verifier passes an episode exactly when its first candidate is its
start. The start check registers a run of the follower as it registers a run of the scripted check, and the check refuses servers
without the AssetOpsBench rows or without the MCP before the run folder exists.
No network, no model."""
import importlib.util
import json
from pathlib import Path

import pytest

from CausalDiagnosis.utils.json_files import read_json, write_json
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.datasets.splits.development_split import DEVELOPMENT_SEED, coverage_order, development_starts, faulty_levels  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.tests.test_run_episodes import _parsed, _rows  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate  # noqa: E402

FOLLOWER = "mcp_follower"
NO_TABLE = f"--agent {FOLLOWER} is a check that no table reads"
REFUSAL = "follows the CausalDiagnosis MCP through the servers of the AssetOpsBench rows"


def _start_cap():
    """--start-cap with the smallest cap that the driver accepts on the development starts of the base row, the starts that cover
    every faulty level, so that a run plays as few episodes as the driver allows."""
    _facts, _cycles, blocks = load_world()
    _a, _name, config = _parsed("--config", "base", "--agent", FOLLOWER)
    healthy = [config.healthy[c] for c in config.components]
    starts = development_starts(run_episodes.get_profile(config.asset_id), set(blocks))
    return ["--start-cap", str(len(coverage_order(starts, lambda start: faulty_levels(start, healthy), DEVELOPMENT_SEED)[0]))]


def test_every_episode_submits_the_first_candidate_orders_its_faulty_components_and_passes_exactly_when_it_is_the_start(tmp_path):
    """Each episode asks the profile and the diagnostics tools for their whole answer, reads the record without a sensor list, so the ranking
    leaves no group to read, submits the levels of the first
    candidate with citations of the newest row that the engine maps onto released values, and orders the child asset of each
    component the candidate puts off its healthy level. The verifier then scores no refusal and no provenance failure, recovers the
    rig exactly when the orders cover the faulty components of the start, counts every other order as unnecessary, and passes the
    episode exactly when the first candidate is the start. The run of development starts carries the stamp development, and the
    declaration of the follower adds no reason to it."""
    run_dir = run_episodes.main(["--config", "base", "--agent", FOLLOWER, "--servers", "aob", "--store", "memory", "--starts", "development",
                                 "--seeds", "1", *_start_cap(), "--no-grade", "--out", str(tmp_path / "out")])
    meta = read_json(run_dir / "run.json")
    assert meta["stamp"] == "development" and NO_TABLE not in meta["stamp_reasons"]
    rows = _rows(run_dir)
    assert rows and len(rows) == meta["episodes_planned"]
    for row in rows:
        episode = run_dir / f"seed{int(row['seed']):06d}"
        steps = read_json(episode / "trajectory.json")
        tools = [step["tool"] for step in steps]
        reads = [step for step in steps if step["tool"] == "history"]
        orders = [step for step in steps if step["tool"] == "generate_work_order"]
        assert tools == ["get_asset_profile", *["history"] * len(reads), "get_episode_evidence", "rank_root_causes", "submit_diagnosis",
                         *["generate_work_order"] * len(orders), "finish"], row["seed"]
        assert all(step["tool_args"].get("response_format") == "raw" for step in steps
                   if step["server"] == "causal_diagnosis" or step["tool"] == "get_asset_profile")
        served = json.loads(steps[0]["response"])
        components, healthy, asset = served["components"], served["healthy"], served["asset"]
        ranking = json.loads(steps[tools.index("rank_root_causes")]["response"])
        first = ranking["candidates"][0]["combination"]
        assert all("sensors" not in step["tool_args"] for step in reads) and set(ranking["left_to_read"].values()) == {0}
        submitted = steps[tools.index("submit_diagnosis")]
        assert {c: submitted["tool_args"][c] for c in components} == first
        newest = max((r for step in reads for r in json.loads(step["response"])["observations"]), key=lambda r: r["timestamp"])
        cited = submitted["tool_args"]["evidence_ids"].split(",")
        assert cited and all(c.startswith(f"obs/{newest['timestamp']}/") for c in cited)
        # the engine echoes a value it mapped by the number of its serving, and an identifier it cannot map as it came
        assert all(e.split("/")[1].isdigit() for e in json.loads(submitted["response"])["evidence_ids"])
        faulty = [c for c in components if first[c] != healthy[c]]
        assert [order["tool_args"]["asset_num"] for order in orders] == [asset["children"][c] for c in faulty]
        assert all(order["tool_args"]["site_id"] == asset["site"] and order["tool_args"]["work_type"] == "CM" and order["tool_args"]["wonum"] == ""
                   for order in orders)
        start = dict(zip(components, json.loads(row["block"])))
        start_faulty = {c for c in components if start[c] != healthy[c]}
        assert (row["provenance_ok"], row["refusals"], row["error"], row["llm_calls"]) == ("True", "0", "", "0"), row["seed"]
        assert row["recovered"] == str(start_faulty <= set(faulty)) and int(row["unnecessary"]) == len(set(faulty) - start_faulty)
        assert row["pass"] == row["diagnosis_all_correct"] == str(first == start), row["seed"]
        assert read_json(episode / "llm_calls.json") == []


def test_a_run_of_the_follower_through_the_driver_enters_policies_and_tab_policies_and_no_table_that_selects_agents(tmp_path):
    """A run of the follower through the driver, read by the tables. Its run folder, with the backend of the store couchdb that its grid
    lines name in place of the memory store of this test, which every table refuses, gives one row of POLICIES on the surface aob over
    every episode it played and its two cells of the base layer in tab:policies, its pass and its reward. Its label enters no other table but RUNS, the index of every
    run folder, and A8 and A12, which group every agent label that answers or ranks with the MCP and which no page and no line of the
    leaderboard reads."""
    run_dir = run_episodes.main(["--config", "base", "--agent", FOLLOWER, "--servers", "aob", "--store", "memory", "--starts", "development",
                                 "--seeds", "1", *_start_cap(), "--no-grade", "--out", str(tmp_path / "runs")])
    meta = read_json(run_dir / "run.json")
    write_json(run_dir / "run.json", {**meta, "backend": {**meta["backend"], "kind": "couchdb"}})
    out = tmp_path / "results" / "tables"
    tables = {t.id: rows for t, rows in aggregate.aggregate(tmp_path / "runs", out, resamples=20, readme_path=out.parent / "README.md",
                                                            published_root=tmp_path / "published", fraction="1")}
    assert {name for name, rows in tables.items() if any(str(value) == FOLLOWER for row in rows for value in row.values())} == {"POLICIES", "RUNS", "A8", "A12"}
    assert [(r["agent"], r["config"], r["surface"], r["n"]) for r in tables["POLICIES"]] == [(FOLLOWER, "base", "aob", meta["episodes_planned"])]
    page = (out.parent / "MAIN_TABLES.md").read_text(encoding="utf-8")
    lines = [line for line in page.split("(`tab:policies`)\n", 1)[1].split("\n#### ")[0].splitlines() if line.startswith("| ")]
    assert lines[0].strip("| ").split(" | ")[-2:] == ["MCP follower pass (%)", "MCP follower reward"]
    base = lines[1].strip("| ").split(" | ")
    assert base[0] == "Base" and "no run" not in base[-2:] and set(base[1:-2]) == {"no run"}


def test_the_start_check_registers_a_run_of_the_follower_and_still_refuses_a_declaration_that_no_table_reads(monkeypatch):
    """With every other rule of the start check passing, a run of the follower on a registered start set of a table preset is of
    record, as the scripted check on the same line is, since POLICIES reads it. The control: the same follower declared with the flag
    registered false carries the reason of its declaration, so the flag still keeps a check that no table reads out of the tables."""
    for function in ("untracked_loaded", "data_reasons"):
        monkeypatch.setattr(run_episodes.ledger.code_identity, function, lambda *args, **kwargs: [])
    monkeypatch.setattr(run_episodes.ledger.code_identity, "open_reasons", lambda: [])
    monkeypatch.setattr(run_episodes.ledger.code_identity, "tree_record_reasons", lambda root=None: [])
    declared = run_episodes.start_check.agent_kinds.DECLARED_AGENTS
    assert "registered" not in declared[FOLLOWER]
    for agent, reasons in (("scripted", []), (FOLLOWER, [])):
        a, name, config = _parsed("--config", "base", "--agent", agent, "--servers", "aob", "--starts", "sampled")
        assert run_episodes.start_check.registration(a, config, name, default_facts_path()) == reasons, agent
    monkeypatch.setitem(declared, FOLLOWER, {**declared[FOLLOWER], "registered": False})
    assert run_episodes.start_check.registration(a, config, name, default_facts_path()) == [NO_TABLE]


@pytest.mark.parametrize("servers", ([], ["--servers", "rig"], ["--servers", "aob", "--no-diagnostics"]))
def test_the_follower_refuses_servers_without_the_rows_or_without_the_mcp_before_its_run_folder(tmp_path, servers):
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=REFUSAL):
        run_episodes.main(["--config", "base", "--agent", FOLLOWER, *servers, "--starts", "development", "--seeds", "1", *_start_cap(),
                           "--no-grade", "--out", str(out)])
    assert not out.exists()
