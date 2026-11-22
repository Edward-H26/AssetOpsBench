"""The MCP follower check of the driver: a policy without a model that plays one episode through the servers of the AssetOpsBench
rows and acts on what the CausalDiagnosis MCP ranks first, so that a run shows whether a pass is reachable through those servers
whatever a model makes of them. It reads the asset profile and every recorded row of the asset, asks the diagnostics server for
the evidence and the ranking, submits the first candidate root cause citing every value of the newest recorded row, orders the
repair of each component that the candidate puts off its healthy level on the child asset of that component at the site of the
asset, and finishes. FOLLOWER is the entry of the declaration (frameworks/AssetOpsBench/__init__.py, AGENTS["mcp_follower"]).
The start check registers a run of it as a run of any declared agent, and of the tables that select agents only POLICIES reads one
(design.POLICY_TABLE_POLICIES), the appendix table of the reference policies.
"""
import json
import time
from types import SimpleNamespace

from CausalDiagnosis.frameworks.AssetOpsBench.agents import upstream_agent_package  # noqa: F401  registers upstream's agent package first
from agent.plan_execute.executor import _ARG_RESOLUTION_PROMPT
from llm import LLMBackend
from CausalDiagnosis.agents.utils.mcp_sessions import open_sessions
from CausalDiagnosis.agents.utils.tool_answers import json_object
from CausalDiagnosis.experiments.episode_runners.run_episodes import diagnostics_calls, run_limited
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, IOT_SERVER, TASK_SERVER, WO_SERVER
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_receipts import split_receipt
from CausalDiagnosis.utils.json_files import write_json


NAME = "mcp_follower"  # the agent kind of the declaration, which the run records as its model id
ORDER_WORK_TYPE = "CM"  # the corrective work type that the briefing of the rows asks of a work order
# the level of an answer that holds every field, which the follower reads off the profile and the ranking
WHOLE_ANSWER = {"response_format": "raw"}


def first_candidate(ranking):
    """The combination that a reply of rank_root_causes puts first, {component: level}. A reply without a candidate, an error
    among them, stops the episode with the reply, since the follower has nothing to submit."""
    candidates = ranking.get("candidates") or []
    if not candidates:
        raise ValueError(f"rank_root_causes named no candidate: {json.dumps(ranking)}")
    return candidates[0]["combination"]


def newest_row_citations(rows):
    """The evidence identifiers of every value of the newest recorded row, obs/<timestamp>/<field> as the row shows them,
    which the metering layer maps onto the values that the engine released."""
    newest = max(rows, key=lambda row: row["timestamp"])
    return [f"obs/{newest['timestamp']}/{field}" for field, value in newest.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)]


def repair_orders(profile, combination):
    """The arguments of generate_work_order for each component that the combination puts off its healthy level, in the order of
    the components of the profile: on the child asset of the component at the site of the asset, with an empty wonum, since the
    environment numbers the orders."""
    asset = profile["asset"]
    return [{"description": f"restore the {component} of {profile['asset_id']} to its healthy level",
             "asset_num": asset["children"][component], "site_id": asset["site"], "work_type": ORDER_WORK_TYPE, "wonum": ""}
            for component in profile["components"] if combination[component] != profile["healthy"][component]]


def closing_summary(combination, orders):
    levels = ", ".join(f"{component} {level}" for component, level in combination.items())
    repaired = ", ".join(order["asset_num"] for order in orders) or "no child asset"
    return f"submitted the first candidate root cause of the CausalDiagnosis MCP, {levels}, and ordered the repair of {repaired}"


async def follow(launchers, trajectory):
    """The episode on one block of sessions, every answered call appended to trajectory in the record of a step of the released
    runner; returns the closing summary. A refusal is an answer, so the record carries no error and the policy goes on, and the
    verifier scores what the refusal left."""
    async with open_sessions(dict(launchers)) as sessions:
        async def call(task, server, tool, arguments):
            started = time.perf_counter()
            text, _is_error = await sessions.call(server, tool, arguments)
            trajectory.append({"step": len(trajectory) + 1, "task": task, "server": server, "tool": tool, "tool_args": arguments,
                               "response": text, "error": None, "duration_ms": (time.perf_counter() - started) * 1000})
            return json_object(split_receipt(text)[0]) or {}

        profile = await call("Read the asset profile", TASK_SERVER, "get_asset_profile", dict(WHOLE_ANSWER))
        asset = {"site_name": profile["asset"]["site"], "asset_id": profile["asset_id"]}
        rows, page = [], {}
        while True:
            # no sensor list, so the read releases every summary of every recorded row, the evidence of the offline table of the
            # MCP on a row without live readings
            page = await call("Read the recorded history of the asset", IOT_SERVER, "history",
                              {**asset, **({"cursor": page["next_cursor"]} if page.get("next_cursor") else {})})
            rows += page.get("observations") or []
            if not (page.get("has_more") and page.get("next_cursor")):
                break
        await call("List the released evidence", CAUSAL_SERVER, "get_episode_evidence", {"sensor_group": "all", **WHOLE_ANSWER})
        combination = first_candidate(await call("Rank the candidate root causes", CAUSAL_SERVER, "rank_root_causes", {"top_k": 1, **WHOLE_ANSWER}))
        await call("Submit the first candidate root cause", TASK_SERVER, "submit_diagnosis",
                   {**combination, "evidence_ids": ",".join(newest_row_citations(rows))})
        orders = repair_orders(profile, combination)
        for order in orders:
            await call(f"Order the repair of {order['asset_num']}", WO_SERVER, "generate_work_order", order)
        summary = closing_summary(combination, orders)
        await call("End the episode", TASK_SERVER, "finish", {"summary": summary})
    return summary


class NoModel(LLMBackend):
    """The backend that the driver records for a check without a model: it names the check and answers no request, since the
    follower sends none."""

    def start_episode(self):
        """Nothing carries from one episode to the next: the policy reads every argument off the answers of its episode."""

    def generate(self, prompt, temperature=0.0):
        raise RuntimeError(f"{NAME} calls no model")

    @property
    def model_id(self):
        return NAME


class FollowerCheck:
    """The follower as the driver asks for a check without a model: the backend of a run on its servers, refused before the run
    folder exists on servers without AssetOpsBench's or without the CausalDiagnosis MCP, and one episode with its record files."""

    # the walk-through reads an argument marker off every check without a model (episode_viewer/walkthrough/call_placement.py);
    # this check calls no model, so the marker of the released executor matches no call of its episodes
    argument_marker = _ARG_RESOLUTION_PROMPT.splitlines()[0]

    def backend(self, servers, profile):
        if IOT_SERVER not in servers or CAUSAL_SERVER not in servers:
            raise SystemExit(f"--agent {NAME} follows the CausalDiagnosis MCP through the servers of the AssetOpsBench rows; "
                             "pass --servers aob and drop --no-diagnostics")
        return NoModel()

    def play(self, runner, launchers, task, run_id, ep_dir):
        """(result, error, extra, calls, tool calls) of one episode, as EpisodeRunner.run_agent returns them. The calls answered
        before an error stay in trajectory.json, so a failed episode is still read call by call."""
        trajectory = []
        summary, error, _ = run_limited(lambda: follow(launchers, trajectory))
        calls = runner.backend.calls
        write_json(ep_dir / "llm_calls.json", calls)
        write_json(ep_dir / "trajectory.json", trajectory)
        (ep_dir / "answer.txt").write_text(summary or "", encoding="utf-8")
        result = None if summary is None else SimpleNamespace(answer=summary, trajectory=trajectory)
        extra = {"llm_calls": len(calls), "input_tokens": sum(c.get("input_tokens", 0) for c in calls),
                 "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
                 "causal_calls": diagnostics_calls((r["server"], r["tool"]) for r in trajectory)}
        tools = [{"server": r["server"], "tool": r["tool"], "error": r["error"]} for r in trajectory]
        return result, error, extra, calls, tools


FOLLOWER = FollowerCheck()
