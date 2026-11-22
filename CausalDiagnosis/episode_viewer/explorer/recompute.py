"""The raw answers of a run that the CausalDiagnosis MCP did not keep (a run folder that holds no transcript.RAW_NAME), recomputed for
the explorer under --hidden alone, since the server of an episode leaves out of its fits the block of the start, which only the
hidden log names. Every call of the CausalDiagnosis MCP is served again in order and at raw, in a folder of its own, by a server
built as the driver builds it for the run (run.json names the ranking model and the abstention rule) over the released evidence
as it stood at the call. That evidence is the final released evidence with the status of the last answer of the action server
before the call and as many of the final events as give the SHA-256 that the line of the call recorded, so no call reads evidence
the agent had not been told. A raw answer counts as recomputed only where the level code of the server writes from it the text
that the agent received; a run whose models.json changed since fails that check and keeps its one level."""
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import build_server
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.answer_levels import at_level
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.mcp_servers.utils.response_levels import text_of

STATUS_KEYS = ("time", "day", "budget_left", "steps_left", "open_work_orders", "episode_over")  # episode_state.write_released


def released_at(final, status, target):
    """The released evidence as the action server wrote it with this status and the first events whose text hashes to target,
    or None."""
    for count in range(len(final["events"]), -1, -1):
        events = final["events"][:count]
        released = {**final, "status": status, "events": events,
                    "observations": [{"n": o["n"], "t": o["t"], "sensor_group": e["sensor_group"], "values": o["values"]}
                                     for e in events if e["event"] == "read" for o in e["served"]]}
        text = json.dumps(released, indent=1)
        if hashlib.sha256(text.encode("utf-8")).hexdigest() == target:
            return text
    return None


async def _served(server, tool, arguments):
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(server._mcp_server) as session:
        answer = await session.call_tool(tool, arguments)
    return "\n".join(block.text for block in answer.content)


def recomputed(folder, calls, run, start, view):
    """{line of the call: the raw answer} for every call of the CausalDiagnosis MCP among calls (the calls of data.episode_calls)
    that kept no raw answer and whose answer the replay wrote again, start the levels of the start in the order of the
    components."""
    folder = Path(folder)
    final = json.loads((folder / "released.json").read_text(encoding="utf-8"))
    kept = transcript.raw_answers(folder)
    status = {"time": 0, **({"budget_left": final["rules"]["budget"]} if final["rules"].get("budget") is not None else {}),
              "steps_left": final["rules"]["max_steps"], "open_work_orders": [], "episode_over": False}
    lines = {line["n"]: line for line in transcript.read(folder) if line.get("kind") == "call"}
    out = {}
    with tempfile.TemporaryDirectory(prefix="explorer_replay_") as tmp:
        released_path = Path(tmp) / "released.json"
        for c in calls:
            answer = c["answer"] if isinstance(c["answer"], dict) else {}
            if c["server"] != CAUSAL_SERVER:
                if answer and "time" in answer and "steps_left" in answer:
                    status = {k: answer[k] for k in STATUS_KEYS if k in answer}
                continue
            line = lines.get(c.get("line"))
            text = line and released_at(final, status, line.get("released_sha256"))
            if text is None:
                continue
            released_path.write_text(text, encoding="utf-8")
            arguments = {**(line.get("arguments") or {}), "response_format": "raw"}
            try:
                server = build_server(released_path, ranking_model=run.get("ranking_model") or "joint_block",
                                      abstention_rule=run.get("abstention_rule") or "decision",
                                      **({"abstention_cut": run["abstention_cut"]} if run.get("abstention_cut") is not None else {}), left_out=tuple(start))
                raw = asyncio.run(_served(server, c["tool"], arguments))  # served in order, so the stored predictions number alike
            except Exception:  # noqa: BLE001  a replay that fails for any reason, as over the fits of another asset, recomputes nothing
                continue
            if c.get("line") in kept or c["level"] == "raw" or "error" in answer or c["level"] is None:
                continue
            seen = {k: v for k, v in answer.items() if k != "unchanged"}
            try:
                again = at_level(c["tool"], json.loads(raw), view, c["level"])
            except (KeyError, TypeError, ValueError):
                continue
            if text_of(again) == text_of(seen):
                out[c["line"]] = raw
    return out
