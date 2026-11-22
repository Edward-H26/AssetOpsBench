"""The walkthrough generator on episodes the drivers write under pytest's tmp_path, outside the repository. A page has the
five parts in order, one step row per tool call and no hidden value before part 3, and for the CausalDiagnosis controller
with a scripted planner every call sits on the plan step that made it, the re-plan is marked and the model calls add up. On
AssetOpsBench's servers the page reads the receipt lines and walks the tools that surface binds. The tests assert relations
between the page and the episode files, never a recorded value."""
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import METERED_SERVERS, RIG_SERVER
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.episode_viewer import walkthrough  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import (ADVISED, SUMMARY_REPLY, ScriptedPlanner, _planner_endpoint, _play, act_now_on_assetopsbench,  # noqa: E402
                                    act_now_with_estimates, plan_json)
from CausalDiagnosis.experiments.episode_runners.tests.test_run_episodes import _run  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.tests.pinned_clock import pinned_preset  # noqa: E402

START, STEPS, HIDDEN_EVENTS, VERDICT, PLAIN = walkthrough.PARTS


def parts_of(text):
    """The text under every second level heading, keyed by the heading, in page order."""
    pieces = text.split("\n## ")[1:]
    return {piece.split("\n", 1)[0]: piece.split("\n", 1)[1] for piece in pieces}


def step_rows(part):
    """The rows of the step table, each a dict from header to cell."""
    lines = part.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Step | Clock after"))
    headers = [h.strip() for h in lines[start].strip().strip("|").split("|")]
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("| "):
            break
        rows.append(dict(zip(headers, (cell.strip() for cell in line.strip().strip("|").split(" | ")))))
    return rows


def test_a_scripted_episode_has_the_five_parts_one_row_per_call_and_no_hidden_value_before_part_3(tmp_path):
    run_dir, rows, _meta = _run(["--config", "static", "--agent", "scripted", "--seeds", "1"], tmp_path / "scripted")
    assert rows[0]["error"] == "", rows[0]["error"]
    ep_dir = run_dir / "seed000000"
    out = tmp_path / "walkthrough.md"
    walkthrough.main([str(ep_dir), "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert walkthrough.FOR_PEOPLE in text.split("\n## ", 1)[0]  # the page says, before its first part, that it shows hidden state
    parts = parts_of(text)
    assert list(parts) == list(walkthrough.PARTS)
    trajectory = read_json(ep_dir / "trajectory.json")
    steps = step_rows(parts[STEPS])
    assert [r["Step"] for r in steps] == [str(n) for n in range(1, len(trajectory) + 1)]
    assert [r["Tool and arguments"].split("`")[1].split("(")[0] for r in steps] == [t["tool"] for t in trajectory]
    # the plan and one argument call per step come before the last tool call, the closing answer after it
    assert steps[-1]["Model calls so far"] == str(len(read_json(ep_dir / "llm_calls.json")) - 1)
    # parts 1, 2 and 5 come from the public files alone, so they read the same without the hidden folder
    bare = tmp_path / "bare" / run_dir.name / ep_dir.name
    shutil.copytree(ep_dir, bare, ignore=shutil.ignore_patterns(HIDDEN))
    for name in ("run.json", "scores.csv"):
        shutil.copy(run_dir / name, bare.parent / name)
    bare_parts = parts_of(walkthrough.render(bare))
    for title in (START, STEPS, PLAIN):
        assert bare_parts[title] == parts[title], title
    assert bare_parts[HIDDEN_EVENTS] != parts[HIDDEN_EVENTS] and bare_parts[VERDICT] != parts[VERDICT]
    # the end of part 5 gives the prices paid against the budget, since the cost with the downtime stays in the hidden log
    envelopes = [json.loads(t["response"]) for t in trajectory if t["server"] == RIG_SERVER]
    budget = read_json(ep_dir / "episode.json")["config"]["budget"]
    paid = walkthrough.num(budget - envelopes[-1]["budget_left"])
    assert f"with {paid} of the budget of {walkthrough.num(budget)} paid in prices." in parts[PLAIN]
    # the hidden start, written the way parts 3 and 4 write it, appears there and nowhere before part 3
    start = walkthrough.long_levels(read_json(ep_dir / HIDDEN / "oracle.json")["diagnosis"], COMPONENTS)
    assert start in parts[HIDDEN_EVENTS] and start in parts[VERDICT] and start not in text[:text.index(f"## {HIDDEN_EVENTS}")]


def test_every_call_of_the_controller_lands_on_its_plan_step_and_the_model_calls_add_up(tmp_path):
    def plans(prompt, call):
        if call == 1:
            # the read of the advised head satisfies the evidence guard, and rank 5 is not on top while rank 1 is, so the diagnosis
            # of the first model plan stops on a surprise before it makes a call
            read = {"kind": "read", "args": {"sensor_group": ADVISED.search(prompt).group(1), "cycles": 1}, "expect": {}, "evidence": []}
            return plan_json([read, {"kind": "diagnose", "args": {"rank": 1}, "expect": {"top_rank": 5}, "evidence": []}])
        if call == 2:
            # a lone wait moves the clock, so the next turn opens with a fresh ranking before its plan
            return plan_json([{"kind": "wait", "args": {"cycles": 1}, "expect": {"no_new_alarm": True}, "evidence": []}])
        return act_now_with_estimates(prompt, call)
    # the fake server calls the evidence insufficient at the first ranking, so the board advises a read batch, which the model reads
    ep_dir, _row, _backend = _play(tmp_path, ScriptedPlanner(plans), {}, force_read_first=1)
    text = walkthrough.render(ep_dir)
    steps = step_rows(parts_of(text)[STEPS])
    trajectory, plans_file = read_json(ep_dir / "trajectory.json"), read_json(ep_dir / "plans.json")
    assert len(steps) == len(trajectory) and all(r["Plan step"] for r in steps)
    # the model plans the opening: plan 1 reads the profile, the failure modes of every component by name and the fix
    # history after its planning call, the orientation follows, and the first ranking comes before plan 2 and before any reading
    components = read_json(ep_dir / "episode.json")["config"]["components"]
    opening = len(components) + 2
    assert [t["tool"] for t in trajectory[:opening]] == ["get_asset_profile", *["get_failure_modes"] * len(components), "get_fix_history"]
    assert [r["Plan step"].startswith("plan 1 step ") for r in steps[:opening]] == [True] * opening
    assert [r["Model calls so far"] for r in steps[:opening]] == ["1"] * opening
    assert steps[opening]["Plan step"] == "before plan 2, the fresh ranking the plan was made from" and steps[opening]["Model calls so far"] == "2"
    # the model plans every read: plan 2 is the model's, its read and the ranking after it come after the planning
    # call, and its diagnosis made no call, which the page says
    first = plans_file["plans"][1]
    assert first["source"] == "model" and first["plan"]["steps"][0]["kind"] == "read"
    read_rows = [r for r in steps if r["Plan step"].startswith("plan 2 step 1")]
    assert len(read_rows) == 2 and {r["Model calls so far"] for r in read_rows} == {"3"}
    surprise, lone_wait = plans_file["plans"][1:3]
    assert surprise["surprise"] is not None and f"Plan {surprise['index']} step 2 made no tool call" in text
    assert lone_wait["after_surprise"] and [r["Plan step"].startswith(f"**re-plan** plan {lone_wait['index']} ") for r in steps].count(True) == 1
    # the first ranking and, since the clock moved during the lone wait, the next turn's fresh ranking sit before their plans and
    # before those plans' model calls
    fresh = [n for n, r in enumerate(steps) if r["Plan step"].startswith("before plan ")]
    assert len(fresh) == 2 and fresh[0] == opening and all(steps[n]["Tool and arguments"].startswith("diagnostics `rank_root_causes(") for n in fresh)
    assert steps[fresh[1]]["Model calls so far"] == steps[fresh[1] - 1]["Model calls so far"]
    for plan in plans_file["plans"]:
        for k, step in enumerate(plan["steps"], 1):
            if step["status"] == "done":
                carried = [r for r in steps if f"plan {plan['index']} step {k}, " in r["Plan step"] and r["Plan step"].endswith(("held", "done"))]
                assert len(carried) == 1 and f"`{walkthrough.RIG_TOOLS[step['kind']]}(" in carried[0]["Tool and arguments"]
    # the closing summary comes right before finish
    calls = read_json(ep_dir / "llm_calls.json")
    assert steps[-1]["Tool and arguments"].startswith("action `finish(") and steps[-1]["Model calls so far"] == str(len(calls))


def test_the_process_of_a_rule_only_episode_names_the_reads_the_rankings_the_rewind_and_the_compactions(tmp_path):
    """The page lists every tool call in order with its price, its cycles and its notifications, the
    ranking and the candidate plans after each reading, every read with the elimination the ranking expected against the one it
    realized, the rewind with the refuted plan and the plan that followed, every compaction with what it collapsed, every
    verification, and the verdict with the truth at the write. The rule only test tool plays the episode without a model."""
    planner = ScriptedPlanner(lambda prompt, call: pytest.fail("the rule only controller must not ask for a plan"))
    ep_dir, _row, _backend = _play(tmp_path, planner, {"rule_only": True}, force_read_first=1, refute_first=True)
    text = walkthrough.render(ep_dir)
    parts = parts_of(text)
    steps = step_rows(parts[STEPS])
    trajectory, plans_file = read_json(ep_dir / "trajectory.json"), read_json(ep_dir / "plans.json")
    assert [r["Step"] for r in steps] == [str(n) for n in range(1, len(trajectory) + 1)]
    # the prices add up to the budget spent, the cycles to the clock of the last action call, and the closed order was notified
    envelopes = [json.loads(t["response"]) for t in trajectory if t["server"] == RIG_SERVER]
    budget = read_json(ep_dir / "episode.json")["config"]["budget"]
    assert sum(float(r["Price"]) for r in steps) == pytest.approx(budget - envelopes[-1]["budget_left"])
    assert sum(int(r["Cycles"]) for r in steps) == envelopes[-1]["time"]
    assert any("Work order" in r["Notifications"] and "closed" in r["Notifications"] for r in steps)
    process = parts[STEPS].split("The controller's process in order")[1]
    ranks = [n for n, t in enumerate(trajectory, 1) if t["tool"] == "rank_root_causes" and json.loads(t["response"]).get("candidates")]
    assert ranks and all(f"- Step {n}, " in process for n in ranks) and ". Candidate plans rank" in process
    # every reading is a read step of a plan, here the rule's, and the read log gives the elimination the ranking expected
    reads = plans_file["board"]["reads"]
    assert reads and all(f"read of {r['sensor_group']} in plan {r['plan']} step {r['step']}" in process for r in reads)
    assert "was expected to take the effective hypotheses from" in process and "eliminated against" in process
    # under the main settings the rule reads what the server advises, and its plan says so
    assert "the server advises this read before a diagnosis" in process and "read batch, plan" not in process
    refuted = plans_file["board"]["refuted"]
    assert len(refuted) == 1 and f"the plan on the {walkthrough.joined(refuted[0]['components'])} was refuted" in process
    rewound = next(p for p in plans_file["plans"] if any((s.get("observed") or {}).get("refuted") for s in p["steps"]))
    # the surprise the reading after the repairs raised is told in the plain words after the orders it followed
    surprise = plans_file["surprises"][-1]
    stop = f"the controller stopped plan {surprise['plan']} at its {surprise['kind']} step and planned again"
    assert stop in parts[PLAIN] and parts[PLAIN].index("submitted the diagnosis") < parts[PLAIN].index(stop)
    assert f"rewind in plan {rewound['index']} step" in process and f"the next plan, plan {rewound['index'] + 1}," in process
    compactions = plans_file["board"]["compaction_log"]
    assert compactions and all(f"- Step {c['after_call']}, compaction after the call," in process for c in compactions)
    verifications = [n for n, t in enumerate(trajectory, 1) if t["tool"] == "verify_outcome"]
    assert verifications and all(f"- Step {n}, verification, " in process for n in verifications)
    verdict = read_json(ep_dir / HIDDEN / "verdict.json")
    if verdict.get("truth_at_write") is not None:
        assert "Truth at the write" in parts[VERDICT] and walkthrough.long_levels(verdict["truth_at_write"], COMPONENTS) in parts[VERDICT]


def test_an_episode_on_the_assetopsbench_real_time_row_is_walked_through_its_receipts(tmp_path, monkeypatch):
    """The controller on AssetOpsBench's iot, fmsr and wo servers and the task server, played by the driver with
    every model call from a local endpoint that hands the prompt to the scripted planner. The page takes the profile from the
    task server, places every call on the plan step that made it, reads the clock, the budget and the notifications from the
    receipt lines and tells the reads, the orders and the commitment from AssetOpsBench's answers instead of quoting them."""
    server, _requests = _planner_endpoint(ScriptedPlanner(act_now_on_assetopsbench))
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        run_dir, _rows, _meta = _run(["--config", pinned_preset(tmp_path, "real_time"), "--agent", "causal", "--model", "vendor/stand-in-1.0",
                                      "--base-url", base, "--seeds", "1", "--servers", "aob"], tmp_path / "aob")
    finally:
        server.shutdown()
    ep_dir = run_dir / "seed000000"
    parts = parts_of(walkthrough.render(ep_dir))
    steps = step_rows(parts[STEPS])
    trajectory = read_json(ep_dir / "trajectory.json")
    assert [r["Step"] for r in steps] == [str(n) for n in range(1, len(trajectory) + 1)]
    assert "It is the answer of `get_asset_profile` at step 1" in parts[START]
    assert "| Failure rule | off, since a patience of 0 cycles switches the rule off |" in parts[START]
    # the walk followed the tools of the AssetOpsBench surface, so every call has its plan step, and the model calls grow to their total
    assert all(r.get("Plan step") for r in steps)
    counts = [int(r["Model calls so far"]) for r in steps]
    assert counts == sorted(counts) and counts[-1] == len(read_json(ep_dir / "llm_calls.json"))
    # the receipts: the prices add up to the budget spent and the cycles to the cycle of the last receipt
    receipts = [json.loads(t["response"].rstrip("\n").rsplit("\n", 1)[1]) for t in trajectory if t["server"] in METERED_SERVERS]
    budget = read_json(ep_dir / "episode.json")["config"]["budget"]
    assert sum(float(r["Price"] or 0) for r in steps) == pytest.approx(budget - receipts[-1]["budget_left"])
    assert sum(int(r["Cycles"] or 0) for r in steps) == receipts[-1]["cycle"] == int(steps[-1]["Clock after"])
    assert any("Work order" in r["Notifications"] and "closed" in r["Notifications"] for r in steps)
    # every answer of AssetOpsBench's servers is told, none quoted, and every history names the group its sensors make up
    aob = [(r, t) for r, t in zip(steps, trajectory) if t["server"] in METERED_SERVERS]
    assert not any(r["Answer"].startswith("answer ") for r, _t in aob)
    groups = read_json(ep_dir / "released.json")["rules"]["sensor_groups"]
    assert all(r["Answer"].split(",")[0] in groups for r, t in aob if t["tool"] == "history")
    # part 3 lists only what the agent was not told, so the alarms that the receipts notified and released.json holds are absent
    released = read_json(ep_dir / "released.json")["events"]
    assert any(e["event"] == "alarm" for e in released) and "| alarm |" not in parts[HIDDEN_EVENTS]
    # part 5 names every order by its wonum and the component of its child asset
    orders = [[json.loads(line) for line in t["response"].rstrip("\n").rsplit("\n", 1)] for t in trajectory if t["tool"] == "generate_work_order"]
    assert orders and all(re.search(rf"{o['wonum']} on the [a-z]+ at cycle {r['cycle']}\b", parts[PLAIN]) for o, r in orders)
    # the real time row has no repair downtime, so the orders close without an executed event, and the iot server serves the rows
    assert "The repairs took no downtime, so the rig never stopped, and the technician closed wo-1" in parts[PLAIN]
    assert "From each closure on, AssetOpsBench's iot server served recordings" in parts[PLAIN]
    assert "A reading takes 0 cycles for each cycle read and a wait 1 cycle for each cycle waited." in parts[START]
    assert "The controller lets a result miss an expectation by the expectation tolerance of the agent options" in parts[STEPS]


def test_part_3_follows_the_hidden_combination_through_drops_and_closes():
    """A close that restored a component names the switch of recordings, a close that left the hidden combination as it was,
    such as an unnecessary order on a healthy component, says so, and a drop between them moves the combination on."""
    components = ["cooler", "valve", "pump", "accumulator"]
    log = [{"event": "reset", "hidden": [20, 100, 0, 100]},
           {"event": "work_order_closed", "t_start": 2, "id": "wo-1", "component": "cooler", "combination_after": [100, 100, 0, 100]},
           {"event": "work_order_closed", "t_start": 3, "id": "wo-2", "component": "valve", "combination_after": [100, 100, 0, 100]},
           {"event": "scheduled_drop", "t_start": 4, "component": "valve", "from_level": 100, "to_level": 90},
           {"event": "work_order_closed", "t_start": 6, "id": "wo-3", "component": "valve", "combination_after": [100, 100, 0, 100]}]
    created = {"wo-1": {"unnecessary": False}, "wo-2": {"unnecessary": True}, "wo-3": {"unnecessary": False}}
    rows = walkthrough.hidden_rows(log, set(), created, components)
    assert [r[0] for r in rows] == ["2", "3", "4", "6"]
    assert rows[0][2] == "the order was necessary, and the rig switched to the recordings of cooler 100, valve 100, pump 0, accumulator 100"
    assert rows[1][2] == ("the order was unnecessary, and the hidden combination stayed cooler 100, valve 100, pump 0, accumulator 100, "
                          "so the rig went on with the same recordings")
    assert rows[3][2].endswith("the rig switched to the recordings of cooler 100, valve 100, pump 0, accumulator 100")


def test_a_count_takes_the_singular_only_when_it_prints_as_1():
    assert walkthrough.plural(0, "cycle") == "0 cycles" and walkthrough.plural(1, "cycle") == "1 cycle"
    assert walkthrough.effective_count(1.0) == "1 effective hypothesis"
    assert walkthrough.effective_count(2.5) == "2.5 effective hypotheses"
    rows = walkthrough.check_rows({"outcome": {"ok": True, "recovered": True, "unnecessary": 1, "unnecessary_orders": ["wo-3"]},
                                   "provenance": {"ok": True, "cited": 0, "unmatched": []}}, ["cooler"])
    assert [r[2] for r in rows] == ["recovered true, 1 unnecessary order wo-3", "no identifier cited"]


def test_a_profile_read_at_the_concise_level_is_shown_whole(tmp_path):
    """The scripted plan on AssetOpsBench's servers asks the task server for the profile without response_format, so it reads the
    default level concise, whose tool table and normal ranges are long tables of columns and rows. The page shows every tool and
    every normal range of those tables as it shows a profile read at raw. The control: a page that reads the concise profile as
    if it were raw fails on the tables."""
    run_dir, rows, _meta = _run(["--config", "base", "--agent", "scripted", "--seeds", "1", "--servers", "aob"], tmp_path / "scripted")
    assert rows[0]["error"] == "", rows[0]["error"]
    ep_dir = run_dir / "seed000000"
    answer = next(t["response"] for t in read_json(ep_dir / "trajectory.json") if t["tool"] == "get_asset_profile")
    profile = json.loads(answer.rstrip("\n").split("\n")[0])
    assert set(profile["tools"]) == {"columns", "rows"} and set(profile["normal_ranges"]) == {"columns", "rows"}  # the concise level
    page = parts_of(walkthrough.render(ep_dir))[START]
    tools = [dict(zip(profile["tools"]["columns"], row)) for row in profile["tools"]["rows"]]
    assert tools and all(f"| `{t['name']}` | {t['server']} | {t['class']} |" in page for t in tools)
    bands = [dict(zip(profile["normal_ranges"]["columns"], row)) for row in profile["normal_ranges"]["rows"]]
    assert bands and all(f"| `{b['field']}` | {b['low']:.8g} | {b['high']:.8g} |" in page for b in bands)


def test_a_reference_policy_page_shows_the_tool_table_its_episode_released(tmp_path):
    """A reference policy receives the profile in the envelope that opens the episode, so the page takes the rules and the
    tool table from released.json, the manifest of the episode's configuration, which on the real time row offers the work
    order without the emergency mode."""
    run_dir, rows, _meta = _run(["--config", pinned_preset(tmp_path, "real_time"), "--agent", "oracle", "--seeds", "1"], tmp_path / "oracle")
    assert rows[0]["error"] == "", rows[0]["error"]
    ep_dir = run_dir / "seed000000"
    parts = parts_of(walkthrough.render(ep_dir))
    assert "The rules and the tool table below are those of `released.json`." in parts[START]
    assert "on the clock rows the answers notify the alarms of the sensor groups already read." in parts[START]
    order = next(t for t in read_json(ep_dir / "released.json")["rules"]["tools"] if t["name"] == "create_work_order")
    row = next(line for line in parts[START].splitlines() if line.startswith("| `create_work_order` |"))
    assert "mode" not in order["parameters"]
    assert row.split(" | ")[3] == ", ".join(f"{name} {kind}" for name, kind in order["parameters"].items())


def test_an_assetopsbench_answer_reads_as_an_envelope_with_or_without_its_receipt():
    """The receipt line of the priced rows gives the clock, the budget and the notifications, the base row's bare text the
    result alone, an error object a refusal, text that holds no JSON object stays text, and the end notification ends the
    episode."""
    body = json.dumps({"wonum": "wo-1"}, indent=2)
    receipt = json.dumps({"price": 40, "budget_left": 260, "steps_left": 50, "cycle": 11, "notifications": [{"event": "end", "t": 11}]})
    envelope = walkthrough.aob_envelope(f"{body}\n{receipt}")
    assert envelope["result"] == {"wonum": "wo-1"} and envelope["time"] == 11 and envelope["budget_left"] == 260 and envelope["episode_over"]
    assert walkthrough.aob_envelope(body) == {"result": {"wonum": "wo-1"}}
    assert walkthrough.aob_envelope(json.dumps({"error": "budget"}))["refused"] == "budget"
    assert walkthrough.aob_envelope("Error executing tool history")["text"] == "Error executing tool history"


def test_a_verification_gives_the_healthy_mass_of_the_belief_now():
    """recovery_posterior of verify_outcome is the mass of the healthy combination in the belief now, which holds every released
    reading and closure and which a ranking on the same evidence reports, so the page names it as the server does."""
    text = walkthrough.diagnostics_answer("verify_outcome", {"prediction_id": "pred/b26", "status": "verified", "consistent": True,
                                                           "recovery_predicted": 0.417, "recovery_posterior": 0.0516}, COMPONENTS)
    assert text == ('pred/b26 status "verified", consistent true, recovery predicted 0.417, '
                    "mass of the healthy combination in the belief now 0.052")


def test_a_verification_names_the_most_probable_combination_after_the_repair_and_its_mass():
    """verify_outcome scores the reading against the most probable combination of the belief after the repair
    and names it with its mass, so the page names both after the healthy mass."""
    text = walkthrough.diagnostics_answer("verify_outcome", {"prediction_id": "pred/b26", "status": "verified", "consistent": True,
                                                           "recovery_predicted": 0.417, "recovery_posterior": 0.0516,
                                                           "predicted_combination": {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130},
                                                           "predicted_mass": 0.912}, COMPONENTS)
    assert text.endswith("mass of the healthy combination in the belief now 0.052, predicted combination "
                         "(100, 100, 0, 130), the most probable after the repair, with mass 0.912")


def test_the_page_names_every_prediction_of_the_mcp_as_a_prediction():
    """The estimate, the plans of a ranking and the expectation of an order name their predicted values as
    predictions, in the words of the server's own texts."""
    estimate = walkthrough.diagnostics_answer("estimate_intervention_effect", {"prediction_id": "pred/1", "action": "replace:pump", "recovery": 0.7,
                                                                             "price": 50, "duration": 5}, COMPONENTS)
    assert "predicted recovery 0.700" in estimate
    branch = walkthrough.branch_text({"faulty_set": ["pump"], "repair_plan": {"steps": [{"component": "pump"}], "price": 50, "cycles_to_restored": 5},
                                      "cost_to_restored": {"act_now": 72, "test_first": 79}, "predicted_outcome": {"recovery_if_acted": 0.6}}, COMPONENTS)
    assert "predicted cost to restored acting now 72 against reading the confirm test and acting on what is first after it 79" in branch
    assert "predicted recovery if acted 0.600" in branch
    assert "expected cost" not in branch and ", recovery if acted" not in branch


def test_part_3_names_the_seconds_behind_the_thinking_cycles_of_a_generation_time_episode(tmp_path):
    """On a row of the generation_time mode the page says how many seconds between the calls the thinking cycles came from, at
    the episode's seconds of thinking per cycle, and how many seconds the state file carried past the last call."""
    run_dir, rows, _meta = _run(["--config", "real_time", "--agent", "oracle", "--seeds", "1"], tmp_path / "oracle")
    assert rows[0]["error"] == "", rows[0]["error"]
    ep_dir = run_dir / "seed000000"
    state = read_json(ep_dir / "episode.json")
    line = next(line for line in parts_of(walkthrough.render(ep_dir))[HIDDEN_EVENTS].splitlines() if line.startswith("The log charges"))
    for value in (sum(state["think_seconds"]), state["config"]["think_seconds_per_cycle"], state["think_seconds_carried"]):
        assert f" {walkthrough.num(value)} seconds" in line, (value, line)


def test_the_run_columns_of_a_gepa_candidate_are_those_of_its_own_row(tmp_path):
    """The page of an episode of a GEPA loop reads the row of its own metric call, which the folder name holds, and not the
    first row of its seed."""
    from types import SimpleNamespace

    from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS
    from CausalDiagnosis.utils.csv_rows import write_rows

    run = tmp_path / "run"
    run.mkdir()
    rows = [{**{c: "" for c in COLUMNS}, "seed": 42, "gepa_call": call, "config": "base", "agent": "causal", "model": f"candidate{call}"}
            for call in (1, 2)]
    write_rows(run / "scores.csv", rows, COLUMNS)
    episode = SimpleNamespace(folder=run / "call00002_seed000042", state={"seed": 42})
    assert walkthrough.Episode._run_columns(episode)["model"] == "candidate2"


def test_the_walkthrough_refuses_an_output_inside_the_fork_or_inside_the_episode_folder(tmp_path):
    """The page quotes readings of single cycles and hidden values, so walkthrough/ refuses an output inside the fork, as the
    viewer and the review pack do, and one inside the episode folder, whose files the run ledger hashes."""
    from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT

    episode = tmp_path / "run" / "seed000001"
    episode.mkdir(parents=True)
    for out, words in ((ROOT / "docs" / "hydraulic_rig" / "walkthroughs" / "refused_page.md", "inside the fork"),
                       (episode / "page.md", "inside the episode folder")):
        with pytest.raises(SystemExit, match=words):
            walkthrough.main([str(episode), "--out", str(out)])
        assert not out.exists()
