"""The notices that arise while the agent's iteration makes no counted call. The driver takes a no_call step for such an
iteration, and the environment keeps the notices of its thinking for the next counted call, so a work order that closes
inside that thinking reaches the board with its cycle, on the AssetOpsBench rows through the receipt of the next call, whose
answers list no open orders, and on the rig through the next envelope. The token clock of the repair rules row charges one
cycle per generated token here, and only the idle turn generates tokens, so the close falls inside the no_call step on every
machine."""
import dataclasses
import json

import pytest
from llm.base import LLMResult

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import (
    CAUSAL_SERVER, HIDDEN, METERED_SERVERS, ORDER_ROW, RIG_SERVER, EpisodeRunner, RecordingBackend, ScriptedPlanner, _in_process,
    _section, _visit_start, advised_read, load_config, load_world, plan_json, run_episodes)
from CausalDiagnosis.agents.utils.tool_answers import json_object
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_receipts import split_receipt
from CausalDiagnosis.utils.json_files import read_json

ROW = "repair_rules_token_clock"
IDLE = "Still weighing the repair."  # the reply of the idle turn, which parses as no plan
READ = {"kind": "read", "args": {"sensor_group": "tier3", "cycles": 1}, "expect": {}, "evidence": []}
FINISH = {"kind": "finish", "args": {}, "expect": {}, "evidence": []}


class ThinkingPlanner(ScriptedPlanner):
    """The scripted planner with the tokens its replies generate: the idle reply generates think tokens, every other none."""

    def __init__(self, plans, think):
        super().__init__(plans)
        self.think = think

    def generate_with_usage(self, prompt, temperature=0.0):
        text = self.generate(prompt, temperature)
        return LLMResult(text=text, output_tokens=self.think if text == IDLE else 0)


def _plans(component, level, server, seen):
    """Diagnose and order the component, then one idle turn, then one read, then read and finish; seen collects the boards of the
    turns after the idle one."""
    state = {"idle": False}

    def plans(prompt, call):
        advised = advised_read(prompt)
        if advised and server:
            return advised
        if "Commitment: none" in prompt:
            head = {"branch": 1 if server else None, "mode": "act_now", "why": "scripted", "steps": [
                {"kind": "diagnose", "args": {"rank": 1}, "expect": {}, "evidence": []},
                {"kind": "order", "args": {"component": component, "prediction_id": "estimate"}, "expect": {}, "evidence": []}]}
            if not server:
                head["candidates"] = [{"levels": {component: level}, "failure_mode_ids": []}]
            return json.dumps(head)
        if not state["idle"]:
            state["idle"] = True
            return IDLE
        seen.append(prompt)
        return plan_json([READ] if len(seen) == 1 else [READ, FINISH], mode="test_first")
    return plans


def _run(tmp_path, surface, server):
    """One episode on the repair rules row with the token clock at one cycle per generated token: the episode folder, the row,
    the order's component and the boards of the turns after the idle one."""
    start, faulty = _visit_start("repair_rules")
    component = faulty[0]
    facts_path, cycles, blocks = load_world()
    _name, config = load_config(ROW)
    level = dict(zip(config.components, start["combination"]))[component]
    config = type(config).from_json(dataclasses.replace(config, think_seconds_per_cycle=config.seconds_per_token, start=start,
                                                        diagnostics_server=server).to_json())
    durations = config.durations
    seen = []
    planner = ThinkingPlanner(_plans(component, level, server, seen), durations["lead_time"][component] + durations["repair_downtime"][component] + 3)
    backend = RecordingBackend(planner)
    options = {"branches": True, "single_hypothesis": False, "plan_length": None, "expectations": True, "rule_only": False}
    ep_dir = tmp_path / "seed000000"
    if surface == "aob":
        servers = (*METERED_SERVERS, CAUSAL_SERVER) if server else METERED_SERVERS
        runner = EpisodeRunner(config, ROW, "causal", servers, cycles, blocks, facts_path, "causal-aob",
                               {"judge": None, "grade": False, "ranking_model": run_episodes.DEFAULT_RANKING_MODEL}, backend=backend, scripted=planner,
                               model=planner.model_id, causal_options=options, store={"kind": "memory"})
    else:
        servers = (RIG_SERVER, CAUSAL_SERVER) if server else (RIG_SERVER,)
        runner = EpisodeRunner(config, ROW, "causal", servers, cycles, blocks, facts_path, "causal-rig",
                               {"judge": None, "grade": False, "ranking_model": run_episodes.DEFAULT_RANKING_MODEL}, backend=backend, scripted=planner,
                               model=planner.model_id, causal_options=options)
    meta = {"config_name": ROW, "agent": "causal", "servers": list(servers), "model": planner.model_id, "clock_mode": config.clock_mode,
            "guidance_level": config.guidance_level}
    if surface == "aob":
        row = runner.play(ep_dir, "seed000000", 0, None, config, meta)
    else:
        with _in_process(ep_dir):
            row = runner.play(ep_dir, "seed000000", 0, None, config, meta)
    return ep_dir, row, component, seen


def _notices(surface, response):
    """The notifications of a recorded answer: the receipt line on the AssetOpsBench rows, the envelope on the rig."""
    data = (split_receipt(response)[1] if surface == "aob" else json_object(response)) or {}
    return data.get("notifications") or []


@pytest.mark.parametrize("server", [True, False], ids=["with_mcp", "without_mcp"])
@pytest.mark.parametrize("surface", ["aob", "rig"])
def test_a_close_inside_a_no_call_step_reaches_the_next_call_and_the_board(tmp_path, surface, server):
    """The order closes inside the thinking of the idle turn's no_call step; the answer of the next counted call carries the
    close with its cycle, the board of the next turn shows the order closed and the close on its timeline, the order keeps
    the cycle of its close, and the agent finishes."""
    ep_dir, row, component, seen = _run(tmp_path, surface, server)
    assert row["error"] == "", row["error"]
    log = read_json(ep_dir / HIDDEN / "log.json")
    close = next(e for e in log if e.get("event") == "work_order_closed")
    no_call = next(e for e in log if e.get("event") == "no_call" and (e.get("think_cycles") or 0) > 0)
    stepped = [e for e in log if e.get("step") is not None and e.get("event") != "end"]
    before = max(e["t_end"] for e in stepped if e["step"] < no_call["step"])
    assert before < close["t_end"] <= no_call["t_end"], (before, close, no_call)
    trajectory = read_json(ep_dir / "trajectory.json")
    first = next(i for i, t in enumerate(trajectory) if t["tool"] in ("history", "read_sensors") and i > next(
        j for j, t in enumerate(trajectory) if t["tool"] in ("generate_work_order", "create_work_order")))
    carried = [(n.get("id"), n.get("t")) for n in _notices(surface, trajectory[first]["response"]) if n.get("event") == "work_order_closed"]
    assert carried == [(close["id"], close["t_end"])], trajectory[first]["response"][-600:]
    board_rows = {r[0]: r[4] for r in ORDER_ROW.findall(_section(seen[1], "## Work orders"))}
    assert board_rows == {close["id"]: "closed"}, board_rows
    assert f"work order wo/{close['id']} on the {component} closed" in _section(seen[1], "## Timeline")
    orders = read_json(ep_dir / "plans.json")["board"]["orders"]
    assert (orders[close["id"]]["status"], orders[close["id"]]["closed_t"]) == ("closed", close["t_end"])
    assert row["end_reason"] == "finish", row["end_reason"]
